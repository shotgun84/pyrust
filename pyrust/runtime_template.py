"""The Rust support library linked into every generated crate.

It gives native code CPython's observable behaviour: exception messages (per
Python version), float repr, exact float `//`/`%`, Mersenne Twister `random`,
insertion-ordered hashed dicts, text files, and Python-style tracebacks with
exit status 1 instead of Rust panics.

`RUNTIME_RS` is specialised for the interpreter running PyRust (message texts
and Unicode tables differ between CPython versions): that interpreter is the
reference the native code must match.
"""

from __future__ import annotations

import sys

_TEMPLATE = r"""#![allow(dead_code, unused_imports, unused_variables, unused_mut, unused_macros)]

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering as AtomicOrdering};
use std::sync::Mutex;

/// Minor version of the CPython this crate must behave like.
pub const PY_MINOR: u32 = @@PY_MINOR@@;

@@UNICODE_TABLES@@

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------
//
// A Python exception is a Rust panic whose message is the exception line
// (`ValueError: ...`). Native programs print it like a CPython traceback and
// exit with status 1; hybrid libraries catch it and re-run the original
// Python function instead.

#[cold]
#[inline(never)]
pub fn py_raise(msg: String) -> ! {
    std::panic::panic_any(msg)
}

/// Raised when a value leaves what native code can represent (64-bit ints,
/// lone surrogates, ...). CPython would keep going; native code stops loudly
/// rather than produce a different answer.
#[cold]
#[inline(never)]
pub fn py_limit(what: &str) -> ! {
    py_raise(format!(
        "OverflowError: {} (PyRust native code is limited to 64-bit integers; run this program with python for exact big-integer results)",
        what
    ))
}

/// A construct native code cannot reproduce exactly at run time.
#[cold]
#[inline(never)]
pub fn py_unsupported(what: &str) -> ! {
    py_raise(format!(
        "NotImplementedError: PyRust native code does not support {}; run this program with python",
        what
    ))
}

pub fn py_msg_zero_div(kind: &str) -> String {
    let m = if PY_MINOR >= 14 {
        "division by zero"
    } else {
        match kind {
            "int_floordiv" | "int_divmod" => "integer division or modulo by zero",
            "int_mod" => "integer modulo by zero",
            "float_floordiv" => "float floor division by zero",
            "float_mod" => if PY_MINOR >= 13 { "float modulo by zero" } else { "float modulo" },
            "float_div" => "float division by zero",
            "float_divmod" => "float divmod()",
            _ => "division by zero",
        }
    };
    format!("ZeroDivisionError: {}", m)
}

fn py_panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "unknown error".to_string()
    }
}

fn py_is_exception_line(msg: &str) -> bool {
    let head = msg.split(':').next().unwrap_or("");
    !head.is_empty()
        && head.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        && head.chars().next().map_or(false, |c| c.is_ascii_uppercase())
        && (head.ends_with("Error") || head.ends_with("Exception") || head.ends_with("Exit")
            || head.ends_with("Interrupt") || head.ends_with("Warning") || head == "StopIteration")
}

/// Map a raw panic message to the Python exception line it stands for.
pub fn py_exception_line(msg: &str) -> String {
    if py_is_exception_line(msg) {
        return msg.to_string();
    }
    if msg.starts_with("attempt to") && msg.contains("overflow") {
        return format!(
            "OverflowError: integer result does not fit in 64 bits ({}) (PyRust native code is limited to 64-bit integers; run this program with python for exact big-integer results)",
            msg
        );
    }
    if msg.contains("divide by zero") || msg.contains("remainder with a divisor of zero") {
        return py_msg_zero_div("");
    }
    if msg.contains("stack overflow") {
        return "RecursionError: maximum recursion depth exceeded".to_string();
    }
    format!("SystemError: PyRust internal error: {}", msg)
}

static PY_SCRIPT: Mutex<String> = Mutex::new(String::new());

// What the program has done so far, to decide whether it can be replayed on
// CPython after hitting a native-only limit (64-bit ints, ...).
static PY_EFFECTS: AtomicBool = AtomicBool::new(false); // files written, stdin read, ...
static PY_NONDET: AtomicBool = AtomicBool::new(false); // clock / unseeded random used
static PY_OUT_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PY_ERR_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PY_REPLAY_WANTED: AtomicBool = AtomicBool::new(false);
static PY_REPLAY: Mutex<Option<(&'static str, &'static str, &'static str)>> = Mutex::new(None);

pub fn py_mark_effect() {
    PY_EFFECTS.store(true, AtomicOrdering::Relaxed);
}

pub fn py_mark_nondet() {
    PY_NONDET.store(true, AtomicOrdering::Relaxed);
}

fn py_is_native_limit(line: &str) -> bool {
    line.contains("PyRust native code")
}

fn py_replay_possible() -> bool {
    !PY_EFFECTS.load(AtomicOrdering::Relaxed)
        && !PY_NONDET.load(AtomicOrdering::Relaxed)
        && PY_REPLAY.lock().map(|r| r.is_some()).unwrap_or(false)
}

/// Re-run the whole program on CPython, hiding the output the native run
/// already produced (the program is deterministic up to that point).
fn py_replay() -> ! {
    let (python, script, pkg_parent) = PY_REPLAY.lock().unwrap().unwrap();
    let skip_out = PY_OUT_BYTES.load(AtomicOrdering::Relaxed) as usize;
    let skip_err = PY_ERR_BYTES.load(AtomicOrdering::Relaxed) as usize;
    let mut args = std::env::args();
    let argv0 = args.next().unwrap_or_default();
    let script_arg = if argv0.ends_with(".py") || argv0.ends_with(".pyr") { argv0 } else { script.to_string() };
    let script_dir = std::path::Path::new(script).parent().map(|p| p.display().to_string()).unwrap_or_default();
    let boot = format!(
        "import sys; sys.path.insert(0, {:?}); from pyrust.hybrid_runner import main; sys.exit(main(sys.argv[1:]))",
        pkg_parent
    );
    let child = std::process::Command::new(python)
        .arg("-c")
        .arg(boot)
        .arg(&script_arg)
        .arg("-")
        .arg(&script_dir)
        .args(args)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(_) => std::process::exit(1),
    };
    fn pump(mut src: impl Read, mut dst: impl Write, mut skip: usize) {
        let mut buf = [0u8; 8192];
        loop {
            let n = match src.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            };
            let mut chunk = &buf[..n];
            if skip > 0 {
                let k = skip.min(chunk.len());
                chunk = &chunk[k..];
                skip -= k;
            }
            if !chunk.is_empty() && dst.write_all(chunk).and_then(|_| dst.flush()).is_err() {
                break;
            }
        }
    }
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let t1 = std::thread::spawn(move || pump(out, io::stdout(), skip_out));
    let t2 = std::thread::spawn(move || pump(err, io::stderr(), skip_err));
    let status = child.wait();
    let _ = t1.join();
    let _ = t2.join();
    match status {
        Ok(st) => match st.code() {
            Some(c) => std::process::exit(c),
            None => std::process::exit(1),
        },
        Err(_) => std::process::exit(1),
    }
}

/// Entry point of a native program: runs `body` with Python semantics for
/// uncaught exceptions, recursion depth and buffered stdout. `replay` is
/// (python executable, script path, directory containing the pyrust package).
pub fn py_run_main(body: fn(), replay: Option<(&'static str, &'static str, &'static str)>) {
    if let Some(a0) = std::env::args().next() {
        *PY_SCRIPT.lock().unwrap_or_else(|e| e.into_inner()) = a0;
    }
    *PY_REPLAY.lock().unwrap_or_else(|e| e.into_inner()) = replay;
    PY_DEPTH_LIMIT.store(PY_NATIVE_RECURSION_LIMIT, AtomicOrdering::Relaxed);
    std::panic::set_hook(Box::new(|info| {
        let text = py_panic_text(info.payload());
        let line = py_exception_line(&text);
        if py_is_native_limit(&line) && py_replay_possible() {
            PY_REPLAY_WANTED.store(true, AtomicOrdering::Relaxed);
            py_flush_files();
            py_flush_stdout_quiet();
            return;
        }
        let script = PY_SCRIPT.lock().map(|s| s.clone()).unwrap_or_default();
        {
            // CPython prints the traceback, then flushes stdout at exit.
            let mut err = io::stderr().lock();
            let _ = writeln!(err, "Traceback (most recent call last):");
            let _ = writeln!(err, "  File \"{}\", in <module>", script);
            let _ = writeln!(err, "{}", line);
            let _ = err.flush();
        }
        py_flush_files();
        py_flush_stdout_quiet();
    }));
    // Deep (but legal) Python recursion must not overflow the native stack.
    let worker = std::thread::Builder::new()
        .name("main".to_string())
        .stack_size(1 << 30)
        .spawn(body)
        .expect("failed to start the main thread");
    let ok = worker.join().is_ok();
    py_flush_files();
    if !ok {
        py_flush_stdout_quiet();
        if PY_REPLAY_WANTED.load(AtomicOrdering::Relaxed) {
            py_replay();
        }
        std::process::exit(1);
    }
    py_exit(0);
}

/// `sys.exit(code)` / normal termination: flush everything first.
pub fn py_exit(code: i32) -> ! {
    py_flush_files();
    if let Err(e) = py_flush_stdout() {
        if e.kind() == io::ErrorKind::BrokenPipe {
            py_broken_pipe_at_exit();
        }
    }
    std::process::exit(code)
}

// ---------------------------------------------------------------------------
// Recursion depth
// ---------------------------------------------------------------------------

/// CPython's default limit: the module frame plus 999 nested calls.
pub const PY_NATIVE_RECURSION_LIMIT: u32 = 1000;
/// Hybrid libraries give native recursion a small budget and let Python take
/// over (and raise RecursionError itself, at the right depth) beyond it.
pub static PY_DEPTH_LIMIT: AtomicU32 = AtomicU32::new(100);

thread_local! {
    static PY_DEPTH: Cell<u32> = const { Cell::new(1) };
}

/// Guard pushed on entry to every compiled Python function.
pub struct PyFrame;

impl PyFrame {
    #[inline(always)]
    pub fn enter() -> PyFrame {
        PY_DEPTH.with(|d| {
            let n = d.get() + 1;
            if n > PY_DEPTH_LIMIT.load(AtomicOrdering::Relaxed) {
                py_raise("RecursionError: maximum recursion depth exceeded".to_string());
            }
            d.set(n);
        });
        PyFrame
    }
}

impl Drop for PyFrame {
    #[inline(always)]
    fn drop(&mut self) {
        PY_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

// ---------------------------------------------------------------------------
// Standard output (block-buffered unless it is a terminal, like CPython)
// ---------------------------------------------------------------------------

static PY_OUT: Mutex<Option<io::BufWriter<io::Stdout>>> = Mutex::new(None);
static PY_OUT_TTY: AtomicBool = AtomicBool::new(false);

fn py_out_lock() -> std::sync::MutexGuard<'static, Option<io::BufWriter<io::Stdout>>> {
    let mut g = PY_OUT.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        let out = io::stdout();
        PY_OUT_TTY.store(out.is_terminal(), AtomicOrdering::Relaxed);
        *g = Some(io::BufWriter::with_capacity(1 << 16, out));
    }
    g
}

#[cold]
fn py_broken_pipe_at_exit() -> ! {
    let mut err = io::stderr().lock();
    if PY_MINOR >= 14 {
        let _ = writeln!(err, "Exception ignored while flushing sys.stdout:");
    } else {
        let _ = writeln!(err, "Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='utf-8'>");
    }
    let _ = writeln!(err, "BrokenPipeError: [Errno 32] Broken pipe");
    std::process::exit(120)
}

#[cold]
fn py_stdout_failed(e: io::Error) -> ! {
    if e.kind() == io::ErrorKind::BrokenPipe {
        // CPython raises BrokenPipeError from print() and then fails to
        // flush stdout again at exit: traceback, then status 120.
        let script = PY_SCRIPT.lock().map(|s| s.clone()).unwrap_or_default();
        {
            let mut err = io::stderr().lock();
            let _ = writeln!(err, "Traceback (most recent call last):");
            let _ = writeln!(err, "  File \"{}\", in <module>", script);
            let _ = writeln!(err, "BrokenPipeError: [Errno 32] Broken pipe");
        }
        py_flush_files();
        py_broken_pipe_at_exit();
    }
    py_raise(format!("OSError: {}", py_os_error_text(&e)))
}

pub fn py_write_str(s: &str) {
    PY_OUT_BYTES.fetch_add(s.len() as u64, AtomicOrdering::Relaxed);
    let mut g = py_out_lock();
    let w = g.as_mut().unwrap();
    if let Err(e) = w.write_all(s.as_bytes()) {
        drop(g);
        py_stdout_failed(e);
    }
    if PY_OUT_TTY.load(AtomicOrdering::Relaxed) && s.contains('\n') {
        if let Err(e) = w.flush() {
            drop(g);
            py_stdout_failed(e);
        }
    }
}

pub fn py_write_fmt(args: fmt::Arguments<'_>, newline: bool) {
    let mut text = match args.as_str() {
        Some(s) => s.to_string(),
        None => fmt::format(args),
    };
    if newline {
        text.push('\n');
    }
    PY_OUT_BYTES.fetch_add(text.len() as u64, AtomicOrdering::Relaxed);
    let mut g = py_out_lock();
    let w = g.as_mut().unwrap();
    let mut res = w.write_all(text.as_bytes());
    if res.is_ok() && newline && PY_OUT_TTY.load(AtomicOrdering::Relaxed) {
        res = w.flush();
    }
    if let Err(e) = res {
        drop(g);
        py_stdout_failed(e);
    }
}

pub fn py_flush_stdout() -> io::Result<()> {
    let mut g = PY_OUT.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_mut() {
        Some(w) => w.flush(),
        None => Ok(()),
    }
}

fn py_flush_stdout_quiet() {
    let _ = py_flush_stdout();
}

#[macro_export]
macro_rules! py_println {
    () => { $crate::pyrust_rt::py_write_str("\n") };
    ($($arg:tt)*) => { $crate::pyrust_rt::py_write_fmt(format_args!($($arg)*), true) };
}

#[macro_export]
macro_rules! py_print {
    ($($arg:tt)*) => { $crate::pyrust_rt::py_write_fmt(format_args!($($arg)*), false) };
}

/// Write to stderr. Like CPython, stdout is *not* flushed first (it stays
/// block-buffered when redirected).
pub fn py_eprint(s: &str) {
    PY_ERR_BYTES.fetch_add(s.len() as u64, AtomicOrdering::Relaxed);
    let mut err = io::stderr().lock();
    let _ = err.write_all(s.as_bytes());
    let _ = err.flush();
}

pub fn py_input(prompt: &str) -> String {
    py_mark_effect();
    if !prompt.is_empty() {
        py_write_str(prompt);
    }
    if let Err(e) = py_flush_stdout() {
        py_stdout_failed(e);
    }
    let mut buf = Vec::new();
    match io::stdin().lock().read_until(b'\n', &mut buf) {
        Ok(0) => py_raise("EOFError: EOF when reading a line".to_string()),
        Ok(_) => {}
        Err(e) => py_raise(format!("OSError: {}", py_os_error_text(&e))),
    }
    let mut s = match String::from_utf8(buf) {
        Ok(s) => s,
        Err(_) => py_raise("UnicodeDecodeError: 'utf-8' codec can't decode input".to_string()),
    };
    if s.ends_with('\n') {
        s.pop();
        if s.ends_with('\r') {
            s.pop();
        }
    }
    s
}

// ---------------------------------------------------------------------------
// repr() / str()
// ---------------------------------------------------------------------------

fn py_in_table(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                Ordering::Less
            } else if lo > cp {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

#[inline]
pub fn py_char_isspace(c: char) -> bool {
    if c.is_ascii() {
        matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f')
    } else {
        py_in_table(PY_UCD_SPACE, c)
    }
}

#[inline]
pub fn py_char_isprintable(c: char) -> bool {
    if c.is_ascii() {
        (' '..='~').contains(&c)
    } else {
        py_in_table(PY_UCD_PRINTABLE, c)
    }
}

pub fn py_char_isdigit(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_digit() } else { py_in_table(PY_UCD_DIGIT, c) }
}

pub fn py_char_isdecimal(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_digit() } else { py_in_table(PY_UCD_DECIMAL, c) }
}

pub fn py_char_isnumeric(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_digit() } else { py_in_table(PY_UCD_NUMERIC, c) }
}

pub fn py_char_isalpha(c: char) -> bool {
    if c.is_ascii() { c.is_ascii_alphabetic() } else { py_in_table(PY_UCD_ALPHA, c) }
}

pub fn py_char_isalnum(c: char) -> bool {
    py_char_isalpha(c) || py_char_isdecimal(c) || py_char_isdigit(c) || py_char_isnumeric(c)
}

fn py_str_all(s: &str, f: fn(char) -> bool) -> bool {
    !s.is_empty() && s.chars().all(f)
}

pub fn py_str_isspace(s: &str) -> bool { py_str_all(s, py_char_isspace) }
pub fn py_str_isdigit(s: &str) -> bool { py_str_all(s, py_char_isdigit) }
pub fn py_str_isdecimal(s: &str) -> bool { py_str_all(s, py_char_isdecimal) }
pub fn py_str_isnumeric(s: &str) -> bool { py_str_all(s, py_char_isnumeric) }
pub fn py_str_isalpha(s: &str) -> bool { py_str_all(s, py_char_isalpha) }
pub fn py_str_isalnum(s: &str) -> bool { py_str_all(s, py_char_isalnum) }

/// `repr(str)`: CPython's quote choice and escapes.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if py_char_isprintable(c) => out.push(c),
            c => {
                let cp = c as u32;
                if cp < 0x100 {
                    out.push_str(&format!("\\x{:02x}", cp));
                } else if cp < 0x10000 {
                    out.push_str(&format!("\\u{:04x}", cp));
                } else {
                    out.push_str(&format!("\\U{:08x}", cp));
                }
            }
        }
    }
    out.push(quote);
    out
}

/// `repr(float)` / `str(float)`: shortest round-trip digits, laid out the
/// way CPython does (fixed notation for exponents -4..16, else scientific).
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let sci = format!("{:e}", f);
    let (mant, exp) = sci.split_once('e').unwrap();
    let mut exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let mut digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    if let Some((d, e)) = py_shortest_tie_even(f.abs(), &digits) {
        digits = d;
        exp = e;
    }
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
    out
}

/// Rust and CPython both print the shortest round-tripping digits, but when
/// the exact value lies exactly halfway between two shortest candidates Rust
/// rounds up and CPython rounds to an even last digit. Detect that tie.
fn py_shortest_tie_even(a: f64, digits: &str) -> Option<(String, i32)> {
    let n = digits.len();
    let ext = format!("{:.*e}", n + 24, a);
    let (em, ee) = ext.split_once('e').unwrap();
    let ed: Vec<u8> = em.bytes().filter(|b| b.is_ascii_digit()).collect();
    if ed[n] != b'5' || ed[n + 1..].iter().any(|&b| b != b'0') {
        return None;
    }
    // Confirm the expansion really stops there (every f64 has a finite one).
    let full = format!("{:.1100e}", a);
    let fd: Vec<u8> = full.split_once('e').unwrap().0.bytes().filter(|b| b.is_ascii_digit()).collect();
    if fd[n] != b'5' || fd[n + 1..].iter().any(|&b| b != b'0') {
        return None;
    }
    let exp: i32 = ee.parse().unwrap();
    let down: Vec<u8> = ed[..n].to_vec();
    if down[n - 1] % 2 == 0 {
        let d = String::from_utf8(down).unwrap();
        let text = format!("{}.{}e{}", &d[..1], &d[1..], exp);
        if text.parse::<f64>().ok() == Some(a) {
            return Some((d.trim_end_matches('0').to_string().max("0".to_string()), exp));
        }
    }
    None
}

/// Kept for generated code: `str(float)` is `repr(float)` in Python 3.
pub fn py_float_str(f: f64) -> String {
    py_float_repr(f)
}

pub trait PyRepr {
    fn py_repr(&self) -> String;
    fn py_str(&self) -> String {
        self.py_repr()
    }
}

pub fn py_repr<T: PyRepr + ?Sized>(v: &T) -> String {
    v.py_repr()
}

pub fn py_str<T: PyRepr + ?Sized>(v: &T) -> String {
    v.py_str()
}

impl PyRepr for i64 {
    fn py_repr(&self) -> String { self.to_string() }
}
impl PyRepr for f64 {
    fn py_repr(&self) -> String { py_float_repr(*self) }
}
impl PyRepr for bool {
    fn py_repr(&self) -> String { (if *self { "True" } else { "False" }).to_string() }
}
impl PyRepr for () {
    fn py_repr(&self) -> String { "None".to_string() }
}
impl PyRepr for String {
    fn py_repr(&self) -> String { py_repr_str(self) }
    fn py_str(&self) -> String { self.clone() }
}
impl PyRepr for str {
    fn py_repr(&self) -> String { py_repr_str(self) }
    fn py_str(&self) -> String { self.to_string() }
}
impl<T: PyRepr + ?Sized> PyRepr for &T {
    fn py_repr(&self) -> String { (**self).py_repr() }
    fn py_str(&self) -> String { (**self).py_str() }
}
impl<T: PyRepr + ?Sized> PyRepr for &mut T {
    fn py_repr(&self) -> String { (**self).py_repr() }
    fn py_str(&self) -> String { (**self).py_str() }
}
impl<T: PyRepr> PyRepr for Vec<T> {
    fn py_repr(&self) -> String {
        let parts: Vec<String> = self.iter().map(|x| x.py_repr()).collect();
        format!("[{}]", parts.join(", "))
    }
}
impl<T: PyRepr> PyRepr for HashSet<T> {
    fn py_repr(&self) -> String {
        if self.is_empty() {
            return "set()".to_string();
        }
        let parts: Vec<String> = self.iter().map(|x| x.py_repr()).collect();
        format!("{{{}}}", parts.join(", "))
    }
}
impl<K: PyKey + PyRepr, V: PyRepr> PyRepr for PyDict<K, V> {
    fn py_repr(&self) -> String {
        let parts: Vec<String> = self.iter().map(|(k, v)| format!("{}: {}", k.py_repr(), v.py_repr())).collect();
        format!("{{{}}}", parts.join(", "))
    }
}

macro_rules! py_repr_tuple {
    ($($name:ident : $idx:tt),+) => {
        impl<$($name: PyRepr),+> PyRepr for ($($name,)+) {
            fn py_repr(&self) -> String {
                let parts: Vec<String> = vec![$(self.$idx.py_repr()),+];
                if parts.len() == 1 {
                    format!("({},)", parts[0])
                } else {
                    format!("({})", parts.join(", "))
                }
            }
        }
    };
}
py_repr_tuple!(A: 0);
py_repr_tuple!(A: 0, B: 1);
py_repr_tuple!(A: 0, B: 1, C: 2);
py_repr_tuple!(A: 0, B: 1, C: 2, D: 3);
py_repr_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
py_repr_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
py_repr_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
py_repr_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);

// ---------------------------------------------------------------------------
// format(): Python's format-spec mini-language
// ---------------------------------------------------------------------------

struct PySpec {
    fill: char,
    fill_given: bool,
    align: Option<char>,
    sign: char,
    z: bool,
    alt: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    ty: Option<char>,
}

fn py_parse_spec(spec: &str, tyname: &str) -> PySpec {
    let bad = || -> ! { py_raise(format!("ValueError: Invalid format specifier '{}' for object of type '{}'", spec, tyname)) };
    let c: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let mut sp = PySpec { fill: ' ', fill_given: false, align: None, sign: '-', z: false, alt: false, width: 0, grouping: None, precision: None, ty: None };
    let is_align = |ch: char| matches!(ch, '<' | '>' | '^' | '=');
    if c.len() >= 2 && is_align(c[1]) {
        sp.fill = c[0];
        sp.fill_given = true;
        sp.align = Some(c[1]);
        i = 2;
    } else if !c.is_empty() && is_align(c[0]) {
        sp.align = Some(c[0]);
        i = 1;
    }
    if i < c.len() && matches!(c[i], '+' | '-' | ' ') {
        sp.sign = c[i];
        i += 1;
    }
    if i < c.len() && c[i] == 'z' {
        sp.z = true;
        i += 1;
    }
    if i < c.len() && c[i] == '#' {
        sp.alt = true;
        i += 1;
    }
    let mut zero_flag = false;
    if i < c.len() && c[i] == '0' {
        // CPython: '0' sets the fill when none was given, and numeric
        // alignment '=' when no alignment was given (see below).
        zero_flag = true;
        if !sp.fill_given {
            sp.fill = '0';
        }
        i += 1;
    }
    let start = i;
    while i < c.len() && c[i].is_ascii_digit() {
        i += 1;
    }
    if i > start {
        sp.width = c[start..i].iter().collect::<String>().parse().unwrap_or_else(|_| bad());
    }
    if i < c.len() && matches!(c[i], ',' | '_') {
        sp.grouping = Some(c[i]);
        i += 1;
    }
    if i < c.len() && c[i] == '.' {
        i += 1;
        let ps = i;
        while i < c.len() && c[i].is_ascii_digit() {
            i += 1;
        }
        if i == ps {
            py_raise("ValueError: Format specifier missing precision".to_string());
        }
        sp.precision = Some(c[ps..i].iter().collect::<String>().parse().unwrap_or_else(|_| bad()));
    }
    if i < c.len() {
        sp.ty = Some(c[i]);
        i += 1;
    }
    if i != c.len() {
        bad();
    }
    if zero_flag && sp.align.is_none() && tyname != "str" {
        sp.align = Some('=');
    }
    sp
}

/// Lay out a formatted number: `int_digits` may receive grouping and
/// zero padding (CPython pads with grouped zeros when fill is '0' and
/// alignment is '='), `rest` is the fraction/exponent/suffix.
fn py_finish_number(neg: bool, sp: &PySpec, prefix: &str, int_digits: &str, rest: &str, every: usize) -> String {
    let sign = py_sign_str(neg, sp);
    if sp.align == Some('=') && sp.fill == '0' {
        let target = sp.width.saturating_sub(sign.chars().count() + prefix.chars().count() + rest.chars().count());
        let mut d = int_digits.to_string();
        loop {
            let grouped = match sp.grouping {
                Some(g) => py_group_digits(&d, g, every),
                None => d.clone(),
            };
            if grouped.chars().count() >= target {
                return format!("{}{}{}{}", sign, prefix, grouped, rest);
            }
            d.insert(0, '0');
        }
    }
    let grouped = match sp.grouping {
        Some(g) => py_group_digits(int_digits, g, every),
        None => int_digits.to_string(),
    };
    py_pad_number(sign, prefix, &format!("{}{}", grouped, rest), sp)
}

fn py_group_digits(digits: &str, sep: char, every: usize) -> String {
    let n = digits.len();
    let mut out = String::with_capacity(n + n / every);
    for (k, ch) in digits.chars().enumerate() {
        if k > 0 && (n - k) % every == 0 {
            out.push(sep);
        }
        out.push(ch);
    }
    out
}

/// Pad `sign + prefix + body` to the spec's width.
fn py_pad_number(sign: &str, prefix: &str, body: &str, sp: &PySpec) -> String {
    let len = sign.chars().count() + prefix.chars().count() + body.chars().count();
    if len >= sp.width {
        return format!("{}{}{}", sign, prefix, body);
    }
    let pad = sp.width - len;
    let fill: String = std::iter::repeat(sp.fill).take(pad).collect();
    match sp.align.unwrap_or('>') {
        '<' => format!("{}{}{}{}", sign, prefix, body, fill),
        '^' => {
            let left: String = std::iter::repeat(sp.fill).take(pad / 2).collect();
            let right: String = std::iter::repeat(sp.fill).take(pad - pad / 2).collect();
            format!("{}{}{}{}{}", left, sign, prefix, body, right)
        }
        '=' => format!("{}{}{}{}", sign, prefix, fill, body),
        _ => format!("{}{}{}{}", fill, sign, prefix, body),
    }
}

fn py_pad_str(body: &str, sp: &PySpec) -> String {
    let len = body.chars().count();
    if len >= sp.width {
        return body.to_string();
    }
    let pad = sp.width - len;
    let fill = |n: usize| -> String { std::iter::repeat(sp.fill).take(n).collect() };
    match sp.align.unwrap_or('<') {
        '>' => format!("{}{}", fill(pad), body),
        '^' => format!("{}{}{}", fill(pad / 2), body, fill(pad - pad / 2)),
        '=' => py_raise("ValueError: '=' alignment not allowed in string format specifier".to_string()),
        _ => format!("{}{}", body, fill(pad)),
    }
}

fn py_sign_str(neg: bool, sp: &PySpec) -> &'static str {
    if neg {
        "-"
    } else {
        match sp.sign {
            '+' => "+",
            ' ' => " ",
            _ => "",
        }
    }
}

pub trait PyFormat {
    fn py_format(&self, spec: &str) -> String;
}

pub fn py_format<T: PyFormat + ?Sized>(v: &T, spec: &str) -> String {
    v.py_format(spec)
}

impl PyFormat for i64 {
    fn py_format(&self, spec: &str) -> String {
        if spec.is_empty() {
            return self.to_string();
        }
        let sp = py_parse_spec(spec, "int");
        match sp.ty {
            Some('e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%') => return (*self as f64).py_format(spec),
            _ => {}
        }
        if sp.precision.is_some() {
            py_raise("ValueError: Precision not allowed in integer format specifier".to_string());
        }
        if sp.z {
            py_raise("ValueError: Negative zero coercion (z) not allowed in integer format specifier".to_string());
        }
        let neg = *self < 0;
        let mag = self.unsigned_abs();
        if sp.ty == Some('n') && sp.grouping.is_some() {
            py_raise(format!("ValueError: Cannot specify '{}' with 'n'.", sp.grouping.unwrap()));
        }
        let (digits, prefix, every) = match sp.ty {
            None | Some('d') | Some('n') => (mag.to_string(), "", 3),
            Some('b') => (format!("{:b}", mag), if sp.alt { "0b" } else { "" }, 4),
            Some('o') => (format!("{:o}", mag), if sp.alt { "0o" } else { "" }, 4),
            Some('x') => (format!("{:x}", mag), if sp.alt { "0x" } else { "" }, 4),
            Some('X') => (format!("{:X}", mag), if sp.alt { "0X" } else { "" }, 4),
            Some('c') => {
                if sp.sign != '-' {
                    py_raise("ValueError: Sign not allowed with integer format specifier 'c'".to_string());
                }
                return py_pad_str(&py_chr(*self), &PySpec { align: Some(sp.align.unwrap_or('>')), ..sp });
            }
            Some(t) => py_raise(format!("ValueError: Unknown format code '{}' for object of type 'int'", t)),
        };
        if sp.grouping == Some(',') && every == 4 {
            py_raise(format!("ValueError: Cannot specify ',' with '{}'.", sp.ty.unwrap()));
        }
        py_finish_number(neg, &sp, prefix, &digits, "", every)
    }
}

fn py_fix_exp(s: &str, upper: bool) -> String {
    // Rust "1.5e3" / "1.5e-3" -> Python "1.5e+03" / "1.5e-03".
    match s.split_once('e') {
        Some((m, e)) => {
            let (sign, digits) = if let Some(d) = e.strip_prefix('-') { ('-', d) } else { ('+', e) };
            let pad = if digits.len() < 2 { "0" } else { "" };
            format!("{}{}{}{}{}", m, if upper { 'E' } else { 'e' }, sign, pad, digits)
        }
        None => s.to_string(),
    }
}

fn py_strip_zeros(s: &str) -> String {
    match s.split_once('e') {
        Some((m, e)) => {
            let m = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
            format!("{}e{}", m, e)
        }
        None => {
            if s.contains('.') {
                s.trim_end_matches('0').trim_end_matches('.').to_string()
            } else {
                s.to_string()
            }
        }
    }
}

/// Python 'g' formatting of a finite non-negative value.
fn py_format_g(a: f64, prec: usize, alt: bool, none_type: bool) -> String {
    let p = if prec == 0 { 1 } else { prec };
    let sci = format!("{:.*e}", p - 1, a);
    let exp: i32 = sci.split_once('e').unwrap().1.parse().unwrap();
    // 'g' switches to exponent form at exp >= p; the default ('') type at
    // exp >= p - 1, because it always shows a digit after the point.
    let limit = if none_type { p as i32 - 1 } else { p as i32 };
    let mut out;
    if exp >= -4 && exp < limit {
        out = format!("{:.*}", (p as i32 - 1 - exp).max(0) as usize, a);
        if !alt {
            out = py_strip_zeros(&out);
        }
        if none_type && !out.contains('.') {
            out.push_str(".0");
        } else if alt && !out.contains('.') {
            out.push('.');
        }
    } else {
        out = sci;
        if !alt {
            out = py_strip_zeros(&out);
        } else if !out.split_once('e').unwrap().0.contains('.') {
            out = out.replacen('e', ".e", 1);
        }
        out = py_fix_exp(&out, false);
    }
    out
}

impl PyFormat for f64 {
    fn py_format(&self, spec: &str) -> String {
        if spec.is_empty() {
            return py_float_repr(*self);
        }
        let sp = py_parse_spec(spec, "float");
        let mut x = *self;
        if sp.z && x == 0.0 {
            x = 0.0;
        }
        let neg = x.is_sign_negative() && !x.is_nan();
        let a = x.abs();
        let upper = matches!(sp.ty, Some('E' | 'F' | 'G'));
        let mut body = if !a.is_finite() {
            let t = if a.is_nan() { "nan" } else { "inf" };
            let t = if upper { t.to_uppercase() } else { t.to_string() };
            if sp.ty == Some('%') { format!("{}%", t) } else { t }
        } else {
            match sp.ty {
                Some('f' | 'F') => {
                    let mut t = format!("{:.*}", sp.precision.unwrap_or(6), a);
                    if sp.alt && !t.contains('.') {
                        t.push('.');
                    }
                    t
                }
                Some('e' | 'E') => {
                    let p = sp.precision.unwrap_or(6);
                    let mut t = format!("{:.*e}", p, a);
                    if sp.alt && p == 0 {
                        t = t.replacen('e', ".e", 1);
                    }
                    py_fix_exp(&t, upper)
                }
                Some('g' | 'G' | 'n') => {
                    let t = py_format_g(a, sp.precision.unwrap_or(6), sp.alt, false);
                    if upper { t.to_uppercase() } else { t }
                }
                Some('%') => {
                    let mut t = format!("{:.*}", sp.precision.unwrap_or(6), a * 100.0);
                    if sp.alt && !t.contains('.') {
                        t.push('.');
                    }
                    t.push('%');
                    t
                }
                None => match sp.precision {
                    Some(p) => py_format_g(a, p, sp.alt, true),
                    None => {
                        let r = py_float_repr(a);
                        if sp.alt && !r.split('e').next().unwrap_or("").contains('.') {
                            r.replacen('e', ".e", 1)
                        } else {
                            r
                        }
                    }
                },
                Some(t) => py_raise(format!("ValueError: Unknown format code '{}' for object of type 'float'", t)),
            }
        };
        if sp.ty == Some('n') && sp.grouping.is_some() {
            py_raise(format!("ValueError: Cannot specify '{}' with 'n'.", sp.grouping.unwrap()));
        }
        let split = body.find(|c: char| !c.is_ascii_digit()).unwrap_or(body.len());
        if split == 0 {
            // inf / nan: no digits to group or zero-pad.
            return py_pad_number(py_sign_str(neg, &sp), "", &body, &sp);
        }
        let rest = body.split_off(split);
        py_finish_number(neg, &sp, "", &body, &rest, 3)
    }
}

impl PyFormat for bool {
    fn py_format(&self, spec: &str) -> String {
        if spec.is_empty() {
            return self.py_str();
        }
        (*self as i64).py_format(spec)
    }
}

impl PyFormat for str {
    fn py_format(&self, spec: &str) -> String {
        if spec.is_empty() {
            return self.to_string();
        }
        let sp = py_parse_spec(spec, "str");
        match sp.ty {
            None | Some('s') => {}
            Some(t) => py_raise(format!("ValueError: Unknown format code '{}' for object of type 'str'", t)),
        }
        if sp.sign != '-' {
            py_raise("ValueError: Sign not allowed in string format specifier".to_string());
        }
        if sp.alt {
            py_raise("ValueError: Alternate form (#) not allowed in string format specifier".to_string());
        }
        if sp.grouping.is_some() {
            py_raise("ValueError: Cannot specify ',' with 's'.".to_string());
        }
        let body: String = match sp.precision {
            Some(p) => self.chars().take(p).collect(),
            None => self.to_string(),
        };
        py_pad_str(&body, &sp)
    }
}

impl PyFormat for String {
    fn py_format(&self, spec: &str) -> String {
        self.as_str().py_format(spec)
    }
}

impl PyFormat for PyValue {
    fn py_format(&self, spec: &str) -> String {
        match self {
            PyValue::Int(i) => i.py_format(spec),
            PyValue::Float(f) => f.py_format(spec),
            PyValue::Bool(b) => b.py_format(spec),
            PyValue::Str(s) => s.py_format(spec),
            other => {
                if spec.is_empty() {
                    other.py_str()
                } else {
                    py_raise(format!("TypeError: unsupported format string passed to {}.__format__", other.type_name()))
                }
            }
        }
    }
}

/// `round(int, ndigits)`: Python keeps an int and rounds half-even.
pub fn py_round_int(x: i64, ndigits: i64) -> i64 {
    if ndigits >= 0 {
        return x;
    }
    if ndigits < -19 {
        return 0;
    }
    let p = 10i128.pow((-ndigits) as u32);
    let v = x as i128;
    let q = v.div_euclid(p);
    let r = v.rem_euclid(p);
    let twice = 2 * r;
    let q = if twice > p || (twice == p && q % 2 != 0) { q + 1 } else { q };
    let out = q * p;
    if out < i64::MIN as i128 || out > i64::MAX as i128 {
        py_limit("round() result does not fit in 64 bits");
    }
    out as i64
}

// ---------------------------------------------------------------------------
// Dynamic values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PyValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    Dict(Vec<(String, PyValue)>),
}

impl Default for PyValue {
    fn default() -> Self {
        PyValue::None
    }
}

/// Numeric view of a value: bool is an int, as in Python.
#[derive(Clone, Copy)]
enum PyNum {
    I(i64),
    F(f64),
}

impl PyValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            PyValue::None => "NoneType",
            PyValue::Bool(_) => "bool",
            PyValue::Int(_) => "int",
            PyValue::Float(_) => "float",
            PyValue::Str(_) => "str",
            PyValue::List(_) => "list",
            PyValue::Dict(_) => "dict",
        }
    }

    fn num(&self) -> Option<PyNum> {
        match self {
            PyValue::Bool(b) => Some(PyNum::I(*b as i64)),
            PyValue::Int(i) => Some(PyNum::I(*i)),
            PyValue::Float(f) => Some(PyNum::F(*f)),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            PyValue::None => false,
            PyValue::Bool(b) => *b,
            PyValue::Int(i) => *i != 0,
            PyValue::Float(f) => *f != 0.0,
            PyValue::Str(s) => !s.is_empty(),
            PyValue::List(v) => !v.is_empty(),
            PyValue::Dict(m) => !m.is_empty(),
        }
    }

    /// The value where a float is required (int/bool widen, like Python).
    pub fn as_f64(&self) -> f64 {
        match self.num() {
            Some(PyNum::I(i)) => i as f64,
            Some(PyNum::F(f)) => f,
            None => py_raise(format!("TypeError: must be real number, not {}", self.type_name())),
        }
    }

    /// The value where an int is required.
    pub fn as_i64(&self) -> i64 {
        match self {
            PyValue::Int(i) => *i,
            PyValue::Bool(b) => *b as i64,
            other => py_raise(format!(
                "TypeError: '{}' object cannot be interpreted as an integer",
                other.type_name()
            )),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            PyValue::Str(s) => s.chars().count(),
            PyValue::List(v) => v.len(),
            PyValue::Dict(m) => m.len(),
            other => py_raise(format!("TypeError: object of type '{}' has no len()", other.type_name())),
        }
    }
}

impl PartialEq for PyValue {
    fn eq(&self, other: &Self) -> bool {
        if let (Some(a), Some(b)) = (self.num(), other.num()) {
            return match (a, b) {
                (PyNum::I(x), PyNum::I(y)) => x == y,
                (PyNum::F(x), PyNum::F(y)) => x == y,
                (PyNum::I(x), PyNum::F(y)) | (PyNum::F(y), PyNum::I(x)) => py_cmp_int_float(x, y) == Some(Ordering::Equal),
            };
        }
        match (self, other) {
            (PyValue::None, PyValue::None) => true,
            (PyValue::Str(a), PyValue::Str(b)) => a == b,
            (PyValue::List(a), PyValue::List(b)) => a == b,
            (PyValue::Dict(a), PyValue::Dict(b)) => {
                a.len() == b.len() && a.iter().all(|(k, v)| b.iter().any(|(ok, ov)| ok == k && ov == v))
            }
            _ => false,
        }
    }
}

impl PartialOrd for PyValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if let (Some(a), Some(b)) = (self.num(), other.num()) {
            return match (a, b) {
                (PyNum::I(x), PyNum::I(y)) => x.partial_cmp(&y),
                (PyNum::F(x), PyNum::F(y)) => x.partial_cmp(&y),
                (PyNum::I(x), PyNum::F(y)) => py_cmp_int_float(x, y),
                (PyNum::F(x), PyNum::I(y)) => py_cmp_int_float(y, x).map(Ordering::reverse),
            };
        }
        match (self, other) {
            (PyValue::Str(a), PyValue::Str(b)) => a.partial_cmp(b),
            (PyValue::List(a), PyValue::List(b)) => a.partial_cmp(b),
            (a, b) => py_raise(format!(
                "TypeError: '<' not supported between instances of '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )),
        }
    }
}

impl PyRepr for PyValue {
    fn py_repr(&self) -> String {
        match self {
            PyValue::None => "None".to_string(),
            PyValue::Bool(b) => b.py_repr(),
            PyValue::Int(i) => i.py_repr(),
            PyValue::Float(f) => f.py_repr(),
            PyValue::Str(s) => py_repr_str(s),
            PyValue::List(items) => items.py_repr(),
            PyValue::Dict(map) => {
                let parts: Vec<String> = map.iter().map(|(k, v)| format!("{}: {}", py_repr_str(k), v.py_repr())).collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }
    fn py_str(&self) -> String {
        match self {
            PyValue::Str(s) => s.clone(),
            other => other.py_repr(),
        }
    }
}

impl fmt::Display for PyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.py_str())
    }
}

pub fn py_value_repr(v: &PyValue) -> String {
    v.py_repr()
}

impl From<()> for PyValue {
    fn from(_: ()) -> Self { PyValue::None }
}
impl From<i64> for PyValue {
    fn from(v: i64) -> Self { PyValue::Int(v) }
}
impl From<f64> for PyValue {
    fn from(v: f64) -> Self { PyValue::Float(v) }
}
impl From<bool> for PyValue {
    fn from(v: bool) -> Self { PyValue::Bool(v) }
}
impl From<String> for PyValue {
    fn from(v: String) -> Self { PyValue::Str(v) }
}
impl From<&str> for PyValue {
    fn from(v: &str) -> Self { PyValue::Str(v.to_string()) }
}
impl<T: Into<PyValue> + Clone> From<Vec<T>> for PyValue {
    fn from(v: Vec<T>) -> Self { PyValue::List(v.into_iter().map(|x| x.into()).collect()) }
}
impl<V: Into<PyValue> + Clone> From<PyDict<String, V>> for PyValue {
    fn from(m: PyDict<String, V>) -> Self {
        PyValue::Dict(m.into_vec().into_iter().map(|(k, v)| (k, v.into())).collect())
    }
}

fn py_unsupported_operands(op: &str, a: &PyValue, b: &PyValue) -> ! {
    py_raise(format!(
        "TypeError: unsupported operand type(s) for {}: '{}' and '{}'",
        op,
        a.type_name(),
        b.type_name()
    ))
}

fn py_checked_int(r: Option<i64>, what: &str) -> i64 {
    match r {
        Some(v) => v,
        None => py_limit(&format!("integer {} result does not fit in 64 bits", what)),
    }
}

impl Add for PyValue {
    type Output = PyValue;
    fn add(self, rhs: PyValue) -> PyValue {
        match (self.num(), rhs.num()) {
            (Some(PyNum::I(a)), Some(PyNum::I(b))) => return PyValue::Int(py_checked_int(a.checked_add(b), "addition")),
            (Some(a), Some(b)) => return PyValue::Float(py_num_f(a) + py_num_f(b)),
            _ => {}
        }
        match (self, rhs) {
            (PyValue::Str(a), PyValue::Str(b)) => PyValue::Str(a + &b),
            (PyValue::List(mut a), PyValue::List(b)) => {
                a.extend(b);
                PyValue::List(a)
            }
            (a, b) => py_unsupported_operands("+", &a, &b),
        }
    }
}

fn py_num_f(n: PyNum) -> f64 {
    match n {
        PyNum::I(i) => i as f64,
        PyNum::F(f) => f,
    }
}

impl Sub for PyValue {
    type Output = PyValue;
    fn sub(self, rhs: PyValue) -> PyValue {
        match (self.num(), rhs.num()) {
            (Some(PyNum::I(a)), Some(PyNum::I(b))) => PyValue::Int(py_checked_int(a.checked_sub(b), "subtraction")),
            (Some(a), Some(b)) => PyValue::Float(py_num_f(a) - py_num_f(b)),
            _ => py_unsupported_operands("-", &self, &rhs),
        }
    }
}

impl Mul for PyValue {
    type Output = PyValue;
    fn mul(self, rhs: PyValue) -> PyValue {
        match (self.num(), rhs.num()) {
            (Some(PyNum::I(a)), Some(PyNum::I(b))) => return PyValue::Int(py_checked_int(a.checked_mul(b), "multiplication")),
            (Some(a), Some(b)) => return PyValue::Float(py_num_f(a) * py_num_f(b)),
            _ => {}
        }
        match (&self, &rhs) {
            (PyValue::Str(s), _) if matches!(rhs.num(), Some(PyNum::I(_))) => PyValue::Str(py_str_repeat(s, rhs.as_i64())),
            (_, PyValue::Str(s)) if matches!(self.num(), Some(PyNum::I(_))) => PyValue::Str(py_str_repeat(s, self.as_i64())),
            (PyValue::List(v), _) if matches!(rhs.num(), Some(PyNum::I(_))) => PyValue::List(py_list_repeat(v, rhs.as_i64())),
            (_, PyValue::List(v)) if matches!(self.num(), Some(PyNum::I(_))) => PyValue::List(py_list_repeat(v, self.as_i64())),
            _ => py_unsupported_operands("*", &self, &rhs),
        }
    }
}

impl Div for PyValue {
    type Output = PyValue;
    fn div(self, rhs: PyValue) -> PyValue {
        match (self.num(), rhs.num()) {
            (Some(PyNum::I(a)), Some(PyNum::I(b))) => PyValue::Float(py_truediv_i64(a, b)),
            (Some(a), Some(b)) => PyValue::Float(py_div_f64(py_num_f(a), py_num_f(b))),
            _ => py_unsupported_operands("/", &self, &rhs),
        }
    }
}

impl Rem for PyValue {
    type Output = PyValue;
    fn rem(self, rhs: PyValue) -> PyValue {
        match (self.num(), rhs.num()) {
            (Some(PyNum::I(a)), Some(PyNum::I(b))) => PyValue::Int(py_mod(a, b)),
            (Some(a), Some(b)) => PyValue::Float(py_mod_f64(py_num_f(a), py_num_f(b))),
            _ => py_unsupported_operands("%", &self, &rhs),
        }
    }
}

impl Neg for PyValue {
    type Output = PyValue;
    fn neg(self) -> PyValue {
        match self.num() {
            Some(PyNum::I(i)) => PyValue::Int(py_checked_int(i.checked_neg(), "negation")),
            Some(PyNum::F(f)) => PyValue::Float(-f),
            None => py_raise(format!("TypeError: bad operand type for unary -: '{}'", self.type_name())),
        }
    }
}

pub fn py_val_floordiv(a: PyValue, b: PyValue) -> PyValue {
    match (a.num(), b.num()) {
        (Some(PyNum::I(x)), Some(PyNum::I(y))) => PyValue::Int(py_floordiv(x, y)),
        (Some(x), Some(y)) => PyValue::Float(py_floordiv_f64(py_num_f(x), py_num_f(y))),
        _ => py_unsupported_operands("//", &a, &b),
    }
}

pub fn py_val_pow(base: PyValue, exp: PyValue) -> PyValue {
    match (base.num(), exp.num()) {
        (Some(PyNum::I(x)), Some(PyNum::I(y))) if y >= 0 => PyValue::Int(py_pow(x, y)),
        (Some(x), Some(y)) => PyValue::Float(py_pow_f64(py_num_f(x), py_num_f(y))),
        _ => py_unsupported_operands("** or pow()", &base, &exp),
    }
}

pub fn py_abs(v: &PyValue) -> PyValue {
    match v {
        PyValue::Int(i) => PyValue::Int(py_checked_int(i.checked_abs(), "abs()")),
        PyValue::Bool(b) => PyValue::Int(*b as i64),
        PyValue::Float(f) => PyValue::Float(f.abs()),
        other => py_raise(format!("TypeError: bad operand type for abs(): '{}'", other.type_name())),
    }
}

/// `int(x)` on a dynamic value.
pub fn py_int_of_value(v: &PyValue) -> i64 {
    match v {
        PyValue::Int(i) => *i,
        PyValue::Bool(b) => *b as i64,
        PyValue::Float(f) => py_int_of_float(*f),
        PyValue::Str(s) => py_parse_int(s),
        other => py_raise(format!(
            "TypeError: int() argument must be a string, a bytes-like object or a real number, not '{}'",
            other.type_name()
        )),
    }
}

/// `float(x)` on a dynamic value.
pub fn py_float_of_value(v: &PyValue) -> f64 {
    match v {
        PyValue::Int(i) => *i as f64,
        PyValue::Bool(b) => *b as i64 as f64,
        PyValue::Float(f) => *f,
        PyValue::Str(s) => py_parse_float(s),
        other => py_raise(format!(
            "TypeError: float() argument must be a string or a real number, not '{}'",
            other.type_name()
        )),
    }
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

/// Exact comparison of an int with a float (Python never rounds the int).
pub fn py_cmp_int_float(i: i64, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    if f >= 9223372036854775808.0 {
        return Some(Ordering::Less);
    }
    if f < -9223372036854775808.0 {
        return Some(Ordering::Greater);
    }
    let t = f.trunc();
    let ti = t as i64; // exact: |t| < 2**63
    match i.cmp(&ti) {
        Ordering::Equal => {
            let frac = f - t;
            if frac > 0.0 {
                Some(Ordering::Less)
            } else if frac < 0.0 {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Equal)
            }
        }
        o => Some(o),
    }
}

pub fn py_lt_if(i: i64, f: f64) -> bool { py_cmp_int_float(i, f) == Some(Ordering::Less) }
pub fn py_le_if(i: i64, f: f64) -> bool { matches!(py_cmp_int_float(i, f), Some(Ordering::Less | Ordering::Equal)) }
pub fn py_gt_if(i: i64, f: f64) -> bool { py_cmp_int_float(i, f) == Some(Ordering::Greater) }
pub fn py_ge_if(i: i64, f: f64) -> bool { matches!(py_cmp_int_float(i, f), Some(Ordering::Greater | Ordering::Equal)) }
pub fn py_eq_if(i: i64, f: f64) -> bool { py_cmp_int_float(i, f) == Some(Ordering::Equal) }

#[inline]
pub fn py_floordiv(a: i64, b: i64) -> i64 {
    if b == 0 {
        py_raise(py_msg_zero_div("int_floordiv"));
    }
    if b == -1 {
        return py_checked_int(a.checked_neg(), "floor division");
    }
    let d = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) { d - 1 } else { d }
}

#[inline]
pub fn py_mod(a: i64, b: i64) -> i64 {
    if b == 0 {
        py_raise(py_msg_zero_div("int_mod"));
    }
    if b == -1 {
        return 0;
    }
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r }
}

pub fn py_divmod(a: i64, b: i64) -> (i64, i64) {
    if b == 0 {
        py_raise(py_msg_zero_div("int_divmod"));
    }
    (py_floordiv(a, b), py_mod(a, b))
}

/// CPython's float_divmod (Objects/floatobject.c), exactly.
fn py_float_divmod_raw(vx: f64, wx: f64) -> (f64, f64) {
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(wx);
    }
    let floordiv;
    if div != 0.0 {
        let mut fd = div.floor();
        if div - fd > 0.5 {
            fd += 1.0;
        }
        floordiv = fd;
    } else {
        floordiv = 0.0f64.copysign(vx / wx);
    }
    (floordiv, m)
}

#[inline]
pub fn py_floordiv_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        py_raise(py_msg_zero_div("float_floordiv"));
    }
    py_float_divmod_raw(a, b).0
}

#[inline]
pub fn py_mod_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        py_raise(py_msg_zero_div("float_mod"));
    }
    // CPython's float_rem.
    let mut m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0.0f64.copysign(b);
    }
    m
}

pub fn py_divmod_f64(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        py_raise(py_msg_zero_div("float_divmod"));
    }
    py_float_divmod_raw(a, b)
}

#[inline]
pub fn py_pow(base: i64, exp: i64) -> i64 {
    if exp < 0 {
        // Python returns a float here; a statically-int expression cannot.
        py_limit("integer ** negative integer gives a float in Python");
    }
    match u32::try_from(exp).ok().and_then(|e| base.checked_pow(e)) {
        Some(v) => v,
        None => py_limit("integer power result does not fit in 64 bits"),
    }
}

#[inline]
pub fn py_lshift(a: i64, b: i64) -> i64 {
    if b < 0 {
        py_raise("ValueError: negative shift count".to_string());
    }
    if a == 0 {
        return 0;
    }
    if b >= 63 || ((a << b) >> b) != a {
        py_limit("left shift result does not fit in 64 bits");
    }
    a << b
}

#[inline]
pub fn py_rshift(a: i64, b: i64) -> i64 {
    if b < 0 {
        py_raise("ValueError: negative shift count".to_string());
    }
    if b >= 63 { if a < 0 { -1 } else { 0 } } else { a >> b }
}

#[inline]
pub fn py_pow_f64(base: f64, exp: f64) -> f64 {
    if base == 0.0 && exp < 0.0 {
        if PY_MINOR >= 14 {
            py_raise("ZeroDivisionError: zero to a negative power".to_string());
        }
        py_raise("ZeroDivisionError: 0.0 cannot be raised to a negative power".to_string());
    }
    if base < 0.0 && exp.is_finite() && exp.fract() != 0.0 {
        py_limit("a negative number to a fractional power is a complex number in Python");
    }
    let r = base.powf(exp);
    if r.is_infinite() && base.is_finite() && exp.is_finite() {
        py_raise("OverflowError: (34, 'Result too large')".to_string());
    }
    r
}

#[inline]
pub fn py_div_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        py_raise(py_msg_zero_div("float_div"));
    }
    a / b
}

/// `int / int`: CPython rounds the exact quotient once.
#[inline]
pub fn py_truediv_i64(a: i64, b: i64) -> f64 {
    const EXACT: i64 = 1 << 53;
    if b == 0 {
        py_raise(py_msg_zero_div("int_truediv"));
    }
    if (-EXACT..=EXACT).contains(&a) && (-EXACT..=EXACT).contains(&b) {
        return (a as f64) / (b as f64);
    }
    py_ratio_to_f64(a as i128, b as i128)
}

/// Correctly rounded num/den for integers (|num| < 2**100, |den| < 2**64).
fn py_ratio_to_f64(num: i128, den: i128) -> f64 {
    if den == 0 {
        py_raise(py_msg_zero_div(""));
    }
    let neg = (num < 0) != (den < 0);
    let n = num.unsigned_abs();
    let d = den.unsigned_abs();
    if n == 0 {
        return if neg { -0.0 } else { 0.0 };
    }
    // Scale so the quotient has 55 significant bits, then round half-even.
    let nbits = 128 - n.leading_zeros() as i32;
    let dbits = 128 - d.leading_zeros() as i32;
    let mut shift = 55 - (nbits - dbits);
    let (mut q, mut rem);
    loop {
        let (nn, dd) = if shift >= 0 { (n << shift as u32, d) } else { (n, d << (-shift) as u32) };
        q = nn / dd;
        rem = nn % dd;
        if q >= (1u128 << 55) {
            shift -= 1;
            continue;
        }
        if q < (1u128 << 54) {
            shift += 1;
            continue;
        }
        break;
    }
    // q has 55 bits: keep 53, round on the low 2 (+ sticky remainder).
    let low = q & 3;
    let mut m = q >> 2;
    let sticky = rem != 0;
    if low > 2 || (low == 2 && (sticky || (m & 1) == 1)) {
        m += 1;
    }
    let v = (m as f64) * 2f64.powi(2 - shift);
    if neg { -v } else { v }
}

fn py_math_err(fname: &str, x: f64) -> ! {
    if PY_MINOR < 14 {
        py_raise("ValueError: math domain error".to_string());
    }
    let msg = match fname {
        "sqrt" => format!("expected a nonnegative input, got {}", py_float_repr(x)),
        "log" | "log10" | "log2" => "expected a positive input".to_string(),
        "asin" | "acos" => format!("expected a number in range from -1 up to 1, got {}", py_float_repr(x)),
        "sin" | "cos" | "tan" => format!("expected a finite input, got {}", py_float_repr(x)),
        _ => "math domain error".to_string(),
    };
    py_raise(format!("ValueError: {}", msg))
}

#[cold]
fn py_math_range() -> ! {
    py_raise("OverflowError: math range error".to_string())
}

pub fn py_math1(name: &str, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    match name {
        "sqrt" => { if x < 0.0 { py_math_err(name, x) } x.sqrt() }
        "log" | "ln" => { if x <= 0.0 { py_math_err("log", x) } x.ln() }
        "log10" => { if x <= 0.0 { py_math_err(name, x) } x.log10() }
        "log2" => { if x <= 0.0 { py_math_err(name, x) } x.log2() }
        "sin" => { if x.is_infinite() { py_math_err(name, x) } x.sin() }
        "cos" => { if x.is_infinite() { py_math_err(name, x) } x.cos() }
        "tan" => { if x.is_infinite() { py_math_err(name, x) } x.tan() }
        "asin" => { if x.abs() > 1.0 { py_math_err(name, x) } x.asin() }
        "acos" => { if x.abs() > 1.0 { py_math_err(name, x) } x.acos() }
        "atan" => x.atan(),
        "tanh" => x.tanh(),
        "fabs" | "abs" => x.abs(),
        "exp" | "sinh" | "cosh" => {
            let r = match name { "exp" => x.exp(), "sinh" => x.sinh(), _ => x.cosh() };
            if r.is_infinite() && x.is_finite() { py_math_range() }
            r
        }
        _ => unreachable!("unknown math function"),
    }
}

/// `math.log(x, base)`.
pub fn py_math_log2args(x: f64, base: f64) -> f64 {
    let num = py_math1("log", x);
    let den = py_math1("log", base);
    py_div_f64(num, den)
}

/// `math.pow(x, y)` (unlike `**`, domain errors are ValueError).
pub fn py_math_pow(x: f64, y: f64) -> f64 {
    let r = x.powf(y);
    if x.is_finite() && y.is_finite() {
        if r.is_nan() {
            py_raise("ValueError: math domain error".to_string());
        }
        if r.is_infinite() {
            if x == 0.0 {
                py_raise("ValueError: math domain error".to_string());
            }
            py_math_range();
        }
    }
    r
}

pub fn py_math_hypot(x: f64, y: f64) -> f64 {
    let r = x.hypot(y);
    if r.is_infinite() && x.is_finite() && y.is_finite() {
        py_math_range();
    }
    r
}

/// float -> int conversion (`int(x)`, `math.floor`, `round`): CPython's
/// errors for NaN/inf; values beyond 64 bits are a native limitation.
pub fn py_float_to_int(x: f64) -> i64 {
    if x.is_nan() {
        py_raise("ValueError: cannot convert float NaN to integer".to_string());
    }
    if x.is_infinite() {
        py_raise("OverflowError: cannot convert float infinity to integer".to_string());
    }
    if !(x >= -9223372036854775808.0 && x < 9223372036854775808.0) {
        py_limit("float is too large for a 64-bit integer");
    }
    x as i64
}

pub fn py_int_of_float(x: f64) -> i64 {
    py_float_to_int(x.trunc())
}

pub fn py_round_bankers(x: f64) -> i64 {
    if !x.is_finite() {
        py_float_to_int(x);
    }
    let f = x.floor();
    let d = x - f;
    let r = if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f / 2.0).floor() * 2.0 == f {
        f
    } else {
        f + 1.0
    };
    py_float_to_int(r)
}

/// Round the exact decimal digits string `int_part.frac_part` at `keep`
/// fractional digits (may be negative), half-even. Returns the rounded value.
fn py_round_decimal(neg: bool, int_part: &str, frac_part: &str, keep: i64) -> f64 {
    let mut digits: Vec<u8> = int_part.bytes().chain(frac_part.bytes()).map(|b| b - b'0').collect();
    let point = int_part.len() as i64; // digits before the decimal point
    let cut = point + keep; // number of digits kept
    if cut < 0 {
        return if neg { -0.0 } else { 0.0 };
    }
    let cut = cut as usize;
    if cut >= digits.len() {
        let s: String = digits.iter().map(|d| (d + b'0') as char).collect();
        let v: f64 = format!("{}e{}", if s.is_empty() { "0" } else { &s }, -(digits.len() as i64 - point))
            .parse()
            .unwrap();
        return if neg { -v } else { v };
    }
    let first_dropped = digits[cut];
    let rest_nonzero = digits[cut + 1..].iter().any(|&d| d != 0);
    let last_kept_odd = cut > 0 && digits[cut - 1] % 2 == 1;
    let round_up = first_dropped > 5 || (first_dropped == 5 && (rest_nonzero || last_kept_odd));
    digits.truncate(cut);
    if round_up {
        let mut i = digits.len();
        loop {
            if i == 0 {
                digits.insert(0, 1);
                break;
            }
            i -= 1;
            if digits[i] == 9 {
                digits[i] = 0;
            } else {
                digits[i] += 1;
                break;
            }
        }
    }
    let carry = digits.len() as i64 - cut as i64;
    let s: String = digits.iter().map(|d| (d + b'0') as char).collect();
    let exp = point + carry - (digits.len() as i64);
    let text = format!("{}e{}", if s.is_empty() { "0".to_string() } else { s }, exp);
    let v: f64 = text.parse().unwrap();
    if neg { -v } else { v }
}

/// `round(x, ndigits)`: CPython rounds the exact binary value half-even.
pub fn py_round_prec(x: f64, ndigits: i64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if ndigits > 323 {
        return x;
    }
    if ndigits < -308 {
        return 0.0f64.copysign(x);
    }
    // Exact decimal expansion of x (every f64 has at most 1074 fraction digits).
    let exact = format!("{:.1074}", x.abs());
    let (ip, fp) = exact.split_once('.').unwrap();
    let r = py_round_decimal(x < 0.0, ip, fp, ndigits);
    if r.is_infinite() {
        py_raise("OverflowError: rounded value too large to represent".to_string());
    }
    r
}

/// `int(str)`: CPython's accepted syntax (whitespace, sign, underscores,
/// Unicode decimal digits); results beyond 64 bits are a native limitation.
pub fn py_parse_int(s: &str) -> i64 {
    let bad = || -> ! { py_raise(format!("ValueError: invalid literal for int() with base 10: {}", py_repr_str(s))) };
    let t = s.trim_matches(|c: char| py_char_isspace(c));
    let (neg, body) = match t.chars().next() {
        Some('-') => (true, &t[1..]),
        Some('+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() {
        bad();
    }
    let mut acc: i128 = 0;
    let mut prev_us = true;
    let mut any = false;
    for c in body.chars() {
        if c == '_' {
            if prev_us {
                bad();
            }
            prev_us = true;
            continue;
        }
        let d = match py_decimal_value(c) {
            Some(d) => d,
            None => bad(),
        };
        prev_us = false;
        any = true;
        acc = acc * 10 + d as i128;
        if acc > (1i128 << 64) {
            py_limit("int() result does not fit in 64 bits");
        }
    }
    if prev_us || !any {
        bad();
    }
    let v = if neg { -acc } else { acc };
    if v < i64::MIN as i128 || v > i64::MAX as i128 {
        py_limit("int() result does not fit in 64 bits");
    }
    v as i64
}

/// Decimal value of a Unicode decimal digit (Nd), if it is one.
fn py_decimal_value(c: char) -> Option<u32> {
    if c.is_ascii_digit() {
        return Some(c as u32 - '0' as u32);
    }
    if !py_char_isdecimal(c) {
        return None;
    }
    // Nd digits come in contiguous runs of ten, each starting at a zero.
    let cp = c as u32;
    let run_start = {
        let mut s = cp;
        while s > 0 && py_char_isdecimal(char::from_u32(s - 1).unwrap_or('x')) {
            s -= 1;
        }
        s
    };
    Some((cp - run_start) % 10)
}

/// `float(str)`: CPython's accepted syntax.
pub fn py_parse_float(s: &str) -> f64 {
    let bad = || -> ! { py_raise(format!("ValueError: could not convert string to float: {}", py_repr_str(s))) };
    let t = s.trim_matches(|c: char| py_char_isspace(c));
    if t.is_empty() {
        bad();
    }
    let lower = t.to_ascii_lowercase();
    let (sign, body) = match lower.chars().next() {
        Some('-') => (-1.0, &lower[1..]),
        Some('+') => (1.0, &lower[1..]),
        _ => (1.0, &lower[..]),
    };
    match body {
        "inf" | "infinity" => return sign * f64::INFINITY,
        "nan" => return f64::NAN.copysign(sign),
        _ => {}
    }
    // Digits (ASCII or Unicode decimal) with single underscores between digits.
    let mut clean = String::with_capacity(t.len());
    let mut prev: Option<char> = None;
    for c in t.chars() {
        if c == '_' {
            match prev {
                Some(p) if p.is_ascii_digit() => {}
                _ => bad(),
            }
            prev = Some('_');
            continue;
        }
        if prev == Some('_') && !(c.is_ascii_digit() || py_char_isdecimal(c)) {
            bad();
        }
        let c2 = if c.is_ascii() { c } else {
            match py_decimal_value(c) {
                Some(d) => char::from_digit(d, 10).unwrap(),
                None => bad(),
            }
        };
        clean.push(c2);
        prev = Some(c2);
    }
    if prev == Some('_') {
        bad();
    }
    let ok_chars = clean.chars().all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'));
    if !ok_chars || clean.contains("..") || clean.ends_with(['e', 'E', '+', '-']) {
        bad();
    }
    match clean.parse::<f64>() {
        Ok(v) => v,
        Err(_) => bad(),
    }
}

/// `max(a, b)` / `min(a, b)`: Python keeps the first argument unless the
/// second compares strictly greater (resp. less), NaN included.
#[inline]
pub fn py_max2<T: PartialOrd>(a: T, b: T) -> T {
    if b > a { b } else { a }
}

#[inline]
pub fn py_min2<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}

fn py_empty_minmax(name: &str) -> ! {
    if PY_MINOR >= 12 {
        py_raise(format!("ValueError: {}() iterable argument is empty", name));
    }
    py_raise(format!("ValueError: {}() arg is an empty sequence", name))
}

pub fn py_max_of<T: PartialOrd + Clone>(items: &[T]) -> T {
    let mut it = items.iter();
    let mut best = match it.next() {
        Some(v) => v,
        None => py_empty_minmax("max"),
    };
    for v in it {
        if v > best {
            best = v;
        }
    }
    best.clone()
}

pub fn py_min_of<T: PartialOrd + Clone>(items: &[T]) -> T {
    let mut it = items.iter();
    let mut best = match it.next() {
        Some(v) => v,
        None => py_empty_minmax("min"),
    };
    for v in it {
        if v < best {
            best = v;
        }
    }
    best.clone()
}

pub fn py_max_str(s: &str) -> String {
    match s.chars().max() {
        Some(c) => c.to_string(),
        None => py_empty_minmax("max"),
    }
}

pub fn py_min_str(s: &str) -> String {
    match s.chars().min() {
        Some(c) => c.to_string(),
        None => py_empty_minmax("min"),
    }
}

/// `sum(list_of_floats)` with an int start (0): CPython's exact algorithm,
/// including the Neumaier compensation added in 3.12. Returns an int when the
/// list is empty, so the result is dynamic.
pub fn py_sum_floats(items: &[f64]) -> PyValue {
    if items.is_empty() {
        return PyValue::Int(0);
    }
    PyValue::Float(py_sum_floats_from(0.0, items, true))
}

/// `sum(floats, start_float)`.
pub fn py_sum_floats_start(start: f64, items: &[f64]) -> f64 {
    py_sum_floats_from(start, items, false)
}

fn py_sum_floats_from(start: f64, items: &[f64], int_start: bool) -> f64 {
    let mut it = items.iter();
    // int 0 + first float goes through generic addition, then the float loop.
    let mut f_result = if int_start {
        match it.next() {
            Some(&x) => 0.0 + x,
            None => return 0.0,
        }
    } else {
        start
    };
    if PY_MINOR < 12 {
        for &x in it {
            f_result += x;
        }
        return f_result;
    }
    let mut c = 0.0f64;
    for &x in it {
        let t = f_result + x;
        if f_result.abs() >= x.abs() {
            c += (f_result - t) + x;
        } else {
            c += (x - t) + f_result;
        }
        f_result = t;
    }
    if c != 0.0 && c.is_finite() {
        f_result += c;
    }
    f_result
}

pub fn py_sum_ints(items: &[i64]) -> i64 {
    let mut acc: i64 = 0;
    for &x in items {
        acc = py_checked_int(acc.checked_add(x), "sum()");
    }
    acc
}

/// `sum()` over dynamic values, starting from int 0.
pub fn py_sum_values(items: &[PyValue]) -> PyValue {
    let mut acc = PyValue::Int(0);
    for v in items {
        acc = acc + v.clone();
    }
    acc
}

pub fn py_hex(n: i64) -> String {
    if n < 0 { format!("-0x{:x}", n.unsigned_abs()) } else { format!("0x{:x}", n) }
}
pub fn py_oct(n: i64) -> String {
    if n < 0 { format!("-0o{:o}", n.unsigned_abs()) } else { format!("0o{:o}", n) }
}
pub fn py_bin(n: i64) -> String {
    if n < 0 { format!("-0b{:b}", n.unsigned_abs()) } else { format!("0b{:b}", n) }
}

pub fn py_chr(i: i64) -> String {
    if !(0..0x110000).contains(&i) {
        py_raise("ValueError: chr() arg not in range(0x110000)".to_string());
    }
    match char::from_u32(i as u32) {
        Some(c) => c.to_string(),
        None => py_limit("lone surrogate characters cannot be represented in native strings"),
    }
}

pub fn py_ord(s: &str) -> i64 {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => c as i64,
        _ => py_raise(format!(
            "TypeError: ord() expected a character, but string of length {} found",
            s.chars().count()
        )),
    }
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

#[inline]
fn py_norm_index(len: usize, idx: i64) -> Option<usize> {
    let len = len as i64;
    let r = if idx < 0 { idx + len } else { idx };
    if r >= 0 && r < len { Some(r as usize) } else { None }
}

#[inline]
pub fn py_index<T: Clone>(items: &[T], idx: i64) -> T {
    match py_norm_index(items.len(), idx) {
        Some(i) => items[i].clone(),
        None => py_raise("IndexError: list index out of range".to_string()),
    }
}

/// `seq[i]` borrowed (nested reads `a[i][j]` without copying the row).
#[inline]
pub fn py_index_ref<T>(items: &[T], idx: i64) -> &T {
    match py_norm_index(items.len(), idx) {
        Some(i) => &items[i],
        None => py_raise("IndexError: list index out of range".to_string()),
    }
}

/// `seq[i]` as a mutable place (nested assignment `a[i][j] = v`).
#[inline]
pub fn py_index_mut<T>(items: &mut [T], idx: i64) -> &mut T {
    match py_norm_index(items.len(), idx) {
        Some(i) => &mut items[i],
        None => py_raise("IndexError: list index out of range".to_string()),
    }
}

#[inline]
pub fn py_set_index<T>(items: &mut [T], idx: i64, val: T) {
    match py_norm_index(items.len(), idx) {
        Some(i) => items[i] = val,
        None => py_raise("IndexError: list assignment index out of range".to_string()),
    }
}

pub fn py_tuple_index_error() -> ! {
    py_raise("IndexError: tuple index out of range".to_string())
}

/// CPython's PySlice_AdjustIndices: (start, stop, step, length).
fn py_slice_indices(len: i64, start: Option<i64>, stop: Option<i64>, step: i64) -> (i64, i64, i64) {
    if step == 0 {
        py_raise("ValueError: slice step cannot be zero".to_string());
    }
    let adjust = |v: Option<i64>, default: i64, lo: i64, hi: i64| -> i64 {
        match v {
            None => default,
            Some(mut x) => {
                if x < 0 {
                    x += len;
                    if x < 0 {
                        x = lo;
                    }
                } else if x >= len {
                    x = hi;
                }
                x
            }
        }
    };
    if step > 0 {
        (adjust(start, 0, 0, len), adjust(stop, len, 0, len), step)
    } else {
        (adjust(start, len - 1, -1, len - 1), adjust(stop, -1, -1, len - 1), step)
    }
}

pub fn py_slice<T: Clone>(items: &[T], start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<T> {
    let (s, e, st) = py_slice_indices(items.len() as i64, start, stop, step);
    let mut out = Vec::new();
    let mut i = s;
    if st > 0 {
        while i < e {
            out.push(items[i as usize].clone());
            i += st;
        }
    } else {
        while i > e {
            out.push(items[i as usize].clone());
            i += st;
        }
    }
    out
}

pub fn py_str_index(s: &str, idx: i64) -> String {
    if s.is_ascii() {
        return match py_norm_index(s.len(), idx) {
            Some(i) => s[i..i + 1].to_string(),
            None => py_raise("IndexError: string index out of range".to_string()),
        };
    }
    let chars: Vec<char> = s.chars().collect();
    match py_norm_index(chars.len(), idx) {
        Some(i) => chars[i].to_string(),
        None => py_raise("IndexError: string index out of range".to_string()),
    }
}

pub fn py_str_slice(s: &str, start: Option<i64>, stop: Option<i64>, step: i64) -> String {
    if s.is_ascii() {
        let (a, e, st) = py_slice_indices(s.len() as i64, start, stop, step);
        if st == 1 {
            return if a < e { s[a as usize..e as usize].to_string() } else { String::new() };
        }
    }
    let chars: Vec<char> = s.chars().collect();
    py_slice(&chars, start, stop, step).into_iter().collect()
}

pub fn py_str_len(s: &str) -> i64 {
    if s.is_ascii() { s.len() as i64 } else { s.chars().count() as i64 }
}

pub fn py_str_repeat(s: &str, n: i64) -> String {
    if n <= 0 { String::new() } else { s.repeat(n as usize) }
}

fn py_char_offset(s: &str, byte_idx: usize) -> i64 {
    s[..byte_idx].chars().count() as i64
}

/// Byte offset of the `n`-th character (clamped), for start/end arguments.
fn py_byte_offset(s: &str, char_idx: i64) -> usize {
    if char_idx <= 0 {
        return 0;
    }
    s.char_indices().nth(char_idx as usize).map(|(b, _)| b).unwrap_or(s.len())
}

pub fn py_str_find(s: &str, pat: &str, start: Option<i64>, end: Option<i64>) -> i64 {
    let n = py_str_len(s);
    let st = match start { Some(v) if v < 0 => (v + n).max(0), Some(v) => v, None => 0 };
    let en = match end { Some(v) if v < 0 => (v + n).max(0), Some(v) => v.min(n), None => n };
    if st > n || st > en {
        return -1;
    }
    let bs = py_byte_offset(s, st);
    let be = py_byte_offset(s, en);
    match s[bs..be].find(pat) {
        Some(i) => py_char_offset(s, bs + i),
        None => -1,
    }
}

pub fn py_str_rfind(s: &str, pat: &str) -> i64 {
    match s.rfind(pat) {
        Some(i) => py_char_offset(s, i),
        None => -1,
    }
}

pub fn py_str_index_of(s: &str, pat: &str) -> i64 {
    let r = py_str_find(s, pat, None, None);
    if r < 0 {
        py_raise("ValueError: substring not found".to_string());
    }
    r
}

pub fn py_str_count(s: &str, pat: &str) -> i64 {
    if pat.is_empty() {
        return py_str_len(s) + 1;
    }
    s.matches(pat).count() as i64
}

pub fn py_strip(s: &str, chars: Option<&str>) -> String {
    match chars {
        None => s.trim_matches(|c: char| py_char_isspace(c)).to_string(),
        Some(set) => s.trim_matches(|c: char| set.contains(c)).to_string(),
    }
}

pub fn py_lstrip(s: &str, chars: Option<&str>) -> String {
    match chars {
        None => s.trim_start_matches(|c: char| py_char_isspace(c)).to_string(),
        Some(set) => s.trim_start_matches(|c: char| set.contains(c)).to_string(),
    }
}

pub fn py_rstrip(s: &str, chars: Option<&str>) -> String {
    match chars {
        None => s.trim_end_matches(|c: char| py_char_isspace(c)).to_string(),
        Some(set) => s.trim_end_matches(|c: char| set.contains(c)).to_string(),
    }
}

/// `str.split()` with no separator.
pub fn py_split_whitespace(s: &str) -> Vec<String> {
    py_split_ws_max(s, -1)
}

pub fn py_split_ws_max(s: &str, maxsplit: i64) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s.trim_start_matches(|c: char| py_char_isspace(c));
    while !rest.is_empty() {
        if maxsplit >= 0 && out.len() as i64 == maxsplit {
            out.push(rest.to_string());
            return out;
        }
        match rest.find(|c: char| py_char_isspace(c)) {
            Some(i) => {
                out.push(rest[..i].to_string());
                rest = rest[i..].trim_start_matches(|c: char| py_char_isspace(c));
            }
            None => {
                out.push(rest.to_string());
                break;
            }
        }
    }
    out
}

pub fn py_split(s: &str, sep: &str, maxsplit: i64) -> Vec<String> {
    if sep.is_empty() {
        py_raise("ValueError: empty separator".to_string());
    }
    if maxsplit < 0 {
        s.split(sep).map(|x| x.to_string()).collect()
    } else {
        s.splitn(maxsplit as usize + 1, sep).map(|x| x.to_string()).collect()
    }
}

pub fn py_splitlines(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\n' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
                out.push(std::mem::take(&mut cur));
            }
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                out.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

pub fn py_str_replace(s: &str, old: &str, new: &str, count: i64) -> String {
    if count < 0 {
        return s.replace(old, new);
    }
    s.replacen(old, new, count as usize)
}

pub fn py_capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
    }
}

pub fn py_removeprefix(s: &str, p: &str) -> String {
    s.strip_prefix(p).unwrap_or(s).to_string()
}

pub fn py_removesuffix(s: &str, p: &str) -> String {
    if p.is_empty() { s.to_string() } else { s.strip_suffix(p).unwrap_or(s).to_string() }
}

/// Length check for `a, b = some_list` (ValueError like CPython).
pub fn py_unpack_check(got: usize, expected: usize) {
    if got > expected {
        if PY_MINOR >= 14 {
            py_raise(format!("ValueError: too many values to unpack (expected {}, got {})", expected, got));
        }
        py_raise(format!("ValueError: too many values to unpack (expected {})", expected));
    }
    if got < expected {
        py_raise(format!("ValueError: not enough values to unpack (expected {}, got {})", expected, got));
    }
}

pub fn py_list_concat<T: Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

pub fn py_list_repeat<T: Clone>(a: &[T], n: i64) -> Vec<T> {
    if n <= 0 {
        return Vec::new();
    }
    let count = n as usize;
    let mut out = Vec::with_capacity(a.len() * count);
    for _ in 0..count {
        out.extend_from_slice(a);
    }
    out
}

pub fn py_range(start: i64, stop: i64, step: i64) -> Vec<i64> {
    if step == 0 {
        py_raise("ValueError: range() arg 3 must not be zero".to_string());
    }
    let mut out = Vec::new();
    let mut cur = start;
    if step > 0 {
        while cur < stop {
            out.push(cur);
            cur = match cur.checked_add(step) { Some(v) => v, None => break };
        }
    } else {
        while cur > stop {
            out.push(cur);
            cur = match cur.checked_add(step) { Some(v) => v, None => break };
        }
    }
    out
}

pub fn py_range_iter(start: i64, stop: i64, step: i64) -> impl Iterator<Item = i64> {
    if step == 0 {
        py_raise("ValueError: range() arg 3 must not be zero".to_string());
    }
    let mut cur = start;
    let mut done = false;
    std::iter::from_fn(move || {
        if done || (step > 0 && cur >= stop) || (step < 0 && cur <= stop) {
            return None;
        }
        let v = cur;
        match cur.checked_add(step) {
            Some(n) => cur = n,
            None => done = true,
        }
        Some(v)
    })
}

pub fn py_list_pop<T>(items: &mut Vec<T>) -> T {
    match items.pop() {
        Some(v) => v,
        None => py_raise("IndexError: pop from empty list".to_string()),
    }
}

pub fn py_list_pop_at<T>(items: &mut Vec<T>, idx: i64) -> T {
    if items.is_empty() {
        py_raise("IndexError: pop from empty list".to_string());
    }
    match py_norm_index(items.len(), idx) {
        Some(i) => items.remove(i),
        None => py_raise("IndexError: pop index out of range".to_string()),
    }
}

pub fn py_list_insert<T>(items: &mut Vec<T>, idx: i64, val: T) {
    let len = items.len() as i64;
    let i = if idx < 0 { (idx + len).max(0) } else { idx.min(len) };
    items.insert(i as usize, val);
}

pub fn py_list_del<T>(items: &mut Vec<T>, idx: i64) {
    match py_norm_index(items.len(), idx) {
        Some(i) => {
            items.remove(i);
        }
        None => py_raise("IndexError: list assignment index out of range".to_string()),
    }
}

pub fn py_list_index<T: PartialEq + PyRepr>(items: &[T], val: &T) -> i64 {
    match items.iter().position(|v| v == val) {
        Some(i) => i as i64,
        None => {
            if PY_MINOR >= 14 {
                py_raise("ValueError: list.index(x): x not in list".to_string());
            }
            py_raise(format!("ValueError: {} is not in list", val.py_repr()))
        }
    }
}

pub fn py_sorted<T: Clone + PartialOrd>(items: &[T]) -> Vec<T> {
    let mut out = items.to_vec();
    py_sort(&mut out);
    out
}

/// Stable sort consulting only `<`, like CPython's list.sort. A hand-rolled
/// merge sort: Rust's own sort may panic when the order is not total (NaN).
pub fn py_sort<T: PartialOrd>(items: &mut Vec<T>) {
    let n = items.len();
    if n < 2 {
        return;
    }
    // With NaN the order is not total and the result depends on Timsort's
    // exact comparison sequence: refuse rather than differ.
    if items.iter().any(|x| x.partial_cmp(x).is_none()) {
        py_unsupported("sorting values that are not totally ordered (such as NaN)");
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut buf: Vec<usize> = vec![0; n];
    let mut width = 1;
    while width < n {
        let mut lo = 0;
        while lo < n {
            let mid = (lo + width).min(n);
            let hi = (lo + 2 * width).min(n);
            let (mut i, mut j, mut k) = (lo, mid, lo);
            while i < mid && j < hi {
                // Take from the right only when strictly smaller: stability.
                if items[idx[j]] < items[idx[i]] {
                    buf[k] = idx[j];
                    j += 1;
                } else {
                    buf[k] = idx[i];
                    i += 1;
                }
                k += 1;
            }
            while i < mid {
                buf[k] = idx[i];
                i += 1;
                k += 1;
            }
            while j < hi {
                buf[k] = idx[j];
                j += 1;
                k += 1;
            }
            lo = hi;
        }
        std::mem::swap(&mut idx, &mut buf);
        width *= 2;
    }
    let mut slots: Vec<Option<T>> = items.drain(..).map(Some).collect();
    for i in idx {
        items.push(slots[i].take().unwrap());
    }
}

pub fn py_list_count<T: Clone + PartialEq>(items: &[T], val: &T) -> i64 {
    items.iter().filter(|v| *v == val).count() as i64
}

pub fn py_list_remove<T: Clone + PartialEq>(items: &mut Vec<T>, val: &T) {
    if let Some(i) = items.iter().position(|v| v == val) {
        items.remove(i);
    } else {
        py_raise("ValueError: list.remove(x): x not in list".to_string());
    }
}

pub fn py_splice<T: Clone>(items: &mut Vec<T>, start: Option<i64>, stop: Option<i64>, new: Vec<T>) {
    let len = items.len() as i64;
    let norm = |bound: i64| -> usize { (if bound < 0 { len + bound } else { bound }).clamp(0, len) as usize };
    let s = norm(start.unwrap_or(0));
    let e = norm(stop.unwrap_or(len));
    let e = e.max(s);
    items.splice(s..e, new);
}

// ---------------------------------------------------------------------------
// random: CPython's Mersenne Twister, bit for bit
// ---------------------------------------------------------------------------

struct PyMt {
    mt: [u32; 624],
    idx: usize,
}

impl PyMt {
    fn new() -> Self {
        let mut m = PyMt { mt: [0; 624], idx: 625 };
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x12345678)
            ^ (std::process::id() as u64) << 32;
        m.seed_u64(seed);
        m
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let n = 624usize;
        let mut k = n.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = n - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x80000000;
        self.idx = 624;
    }

    fn seed_u64(&mut self, n: u64) {
        let lo = n as u32;
        let hi = (n >> 32) as u32;
        if hi == 0 {
            self.init_by_array(&[lo]);
        } else {
            self.init_by_array(&[lo, hi]);
        }
    }

    fn genrand_u32(&mut self) -> u32 {
        const MAG01: [u32; 2] = [0, 0x9908b0df];
        if self.idx >= 624 {
            let mt = &mut self.mt;
            let mut kk = 0;
            while kk < 624 - 397 {
                let y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
                mt[kk] = mt[kk + 397] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                kk += 1;
            }
            while kk < 623 {
                let y = (mt[kk] & 0x80000000) | (mt[kk + 1] & 0x7fffffff);
                mt[kk] = mt[kk + 397 - 624] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
                kk += 1;
            }
            let y = (mt[623] & 0x80000000) | (mt[0] & 0x7fffffff);
            mt[623] = mt[396] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        let lo = self.genrand_u32() as u64;
        let hi = (self.genrand_u32() >> (64 - k)) as u64;
        lo | (hi << 32)
    }

    fn randbelow(&mut self, n: u64) -> u64 {
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }
}

thread_local! {
    static PY_RNG: RefCell<Option<PyMt>> = const { RefCell::new(None) };
}

fn py_with_rng<R>(f: impl FnOnce(&mut PyMt) -> R) -> R {
    PY_RNG.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            py_mark_nondet(); // seeded from the clock, like CPython's urandom seed
            *slot = Some(PyMt::new());
        }
        f(slot.as_mut().unwrap())
    })
}

pub fn py_seed_int(seed: i64) {
    PY_RNG.with(|cell| {
        let mut m = PyMt { mt: [0; 624], idx: 625 };
        m.seed_u64(seed.unsigned_abs());
        *cell.borrow_mut() = Some(m);
    });
}

pub fn py_seed() {
    py_mark_nondet();
    py_with_rng(|m| *m = PyMt::new());
}

pub fn py_random() -> f64 {
    py_with_rng(|m| m.random())
}

pub fn py_randint(a: i64, b: i64) -> i64 {
    py_randrange(a, py_checked_int(b.checked_add(1), "randint()"))
}

pub fn py_randrange(start: i64, stop: i64) -> i64 {
    let width = stop as i128 - start as i128;
    if width <= 0 {
        py_raise(format!("ValueError: empty range in randrange({}, {})", start, stop));
    }
    let r = py_with_rng(|m| m.randbelow(width as u64));
    (start as i128 + r as i128) as i64
}

pub fn py_uniform(a: f64, b: f64) -> f64 {
    a + (b - a) * py_random()
}

pub fn py_choice<T: Clone>(items: &[T]) -> T {
    if items.is_empty() {
        py_raise("IndexError: Cannot choose from an empty sequence".to_string());
    }
    let i = py_with_rng(|m| m.randbelow(items.len() as u64));
    items[i as usize].clone()
}

pub fn py_shuffle<T>(items: &mut Vec<T>) {
    let n = items.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = py_with_rng(|m| m.randbelow(i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// json
// ---------------------------------------------------------------------------

pub trait PyJson {
    fn py_json(&self) -> String;
}

fn py_json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
        }
    }
    out.push('"');
    out
}

fn py_json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_string()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        py_float_repr(f)
    }
}

impl PyJson for i64 {
    fn py_json(&self) -> String { self.to_string() }
}
impl PyJson for f64 {
    fn py_json(&self) -> String { py_json_float(*self) }
}
impl PyJson for bool {
    fn py_json(&self) -> String { (if *self { "true" } else { "false" }).to_string() }
}
impl PyJson for () {
    fn py_json(&self) -> String { "null".to_string() }
}
impl PyJson for String {
    fn py_json(&self) -> String { py_json_escape(self) }
}
impl PyJson for str {
    fn py_json(&self) -> String { py_json_escape(self) }
}
impl<T: PyJson> PyJson for Vec<T> {
    fn py_json(&self) -> String {
        let parts: Vec<String> = self.iter().map(|x| x.py_json()).collect();
        format!("[{}]", parts.join(", "))
    }
}

/// How a dict key appears in JSON (Python converts keys to strings).
pub trait PyJsonKey {
    fn py_json_key(&self) -> String;
}
impl PyJsonKey for String {
    fn py_json_key(&self) -> String { py_json_escape(self) }
}
impl PyJsonKey for i64 {
    fn py_json_key(&self) -> String { format!("\"{}\"", self) }
}
impl PyJsonKey for bool {
    fn py_json_key(&self) -> String { (if *self { "\"true\"" } else { "\"false\"" }).to_string() }
}
impl PyJsonKey for f64 {
    fn py_json_key(&self) -> String { format!("\"{}\"", py_json_float(*self)) }
}

impl<K: PyJsonKey + PyKey, V: PyJson> PyJson for PyDict<K, V> {
    fn py_json(&self) -> String {
        let parts: Vec<String> = self.iter().map(|(k, v)| format!("{}: {}", k.py_json_key(), v.py_json())).collect();
        format!("{{{}}}", parts.join(", "))
    }
}

impl PyJson for PyValue {
    fn py_json(&self) -> String {
        match self {
            PyValue::None => "null".to_string(),
            PyValue::Bool(b) => b.py_json(),
            PyValue::Int(i) => i.py_json(),
            PyValue::Float(f) => f.py_json(),
            PyValue::Str(s) => s.py_json(),
            PyValue::List(items) => items.py_json(),
            PyValue::Dict(map) => {
                let parts: Vec<String> = map.iter().map(|(k, v)| format!("{}: {}", py_json_escape(k), v.py_json())).collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }
}

macro_rules! py_json_tuple {
    ($($name:ident : $idx:tt),+) => {
        impl<$($name: PyJson),+> PyJson for ($($name,)+) {
            fn py_json(&self) -> String {
                let parts: Vec<String> = vec![$(self.$idx.py_json()),+];
                format!("[{}]", parts.join(", "))
            }
        }
    };
}
py_json_tuple!(A: 0);
py_json_tuple!(A: 0, B: 1);
py_json_tuple!(A: 0, B: 1, C: 2);
py_json_tuple!(A: 0, B: 1, C: 2, D: 3);

pub fn py_json_dumps<T: PyJson + ?Sized>(v: &T) -> String {
    v.py_json()
}

struct JsonParser<'a> {
    chars: Vec<char>,
    pos: usize,
    src: &'a str,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        JsonParser { chars: src.chars().collect(), pos: 0, src }
    }

    fn err_at(&self, msg: &str, pos: usize) -> ! {
        let before: String = self.chars[..pos.min(self.chars.len())].iter().collect();
        let lineno = before.matches('\n').count() + 1;
        let colno = match before.rfind('\n') {
            Some(i) => before[i + 1..].chars().count() + 1,
            None => pos + 1,
        };
        py_raise(format!("JSONDecodeError: {}: line {} column {} (char {})", msg, lineno, colno, pos))
    }

    fn ws(&mut self) {
        while self.pos < self.chars.len() && matches!(self.chars[self.pos], ' ' | '\t' | '\n' | '\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn starts_with(&self, lit: &str) -> bool {
        let n = lit.chars().count();
        self.pos + n <= self.chars.len() && self.chars[self.pos..self.pos + n].iter().copied().eq(lit.chars())
    }

    fn parse_document(&mut self) -> PyValue {
        self.ws();
        let v = self.parse_value();
        self.ws();
        if self.pos != self.chars.len() {
            self.err_at("Extra data", self.pos);
        }
        v
    }

    fn parse_value(&mut self) -> PyValue {
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => {
                let start = self.pos;
                self.pos += 1;
                PyValue::Str(self.parse_string(start))
            }
            Some('n') if self.starts_with("null") => { self.pos += 4; PyValue::None }
            Some('t') if self.starts_with("true") => { self.pos += 4; PyValue::Bool(true) }
            Some('f') if self.starts_with("false") => { self.pos += 5; PyValue::Bool(false) }
            Some('N') if self.starts_with("NaN") => { self.pos += 3; PyValue::Float(f64::NAN) }
            Some('I') if self.starts_with("Infinity") => { self.pos += 8; PyValue::Float(f64::INFINITY) }
            Some('-') if self.starts_with("-Infinity") => { self.pos += 9; PyValue::Float(f64::NEG_INFINITY) }
            Some(c) if c == '-' || c.is_ascii_digit() => match self.parse_number() {
                Some(v) => v,
                None => self.err_at("Expecting value", self.pos),
            },
            _ => self.err_at("Expecting value", self.pos),
        }
    }

    fn parse_object(&mut self) -> PyValue {
        self.pos += 1;
        let mut map: Vec<(String, PyValue)> = Vec::new();
        self.ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return PyValue::Dict(map);
        }
        loop {
            if self.peek() != Some('"') {
                self.err_at("Expecting property name enclosed in double quotes", self.pos);
            }
            let start = self.pos;
            self.pos += 1;
            let k = self.parse_string(start);
            self.ws();
            if self.peek() != Some(':') {
                self.err_at("Expecting ':' delimiter", self.pos);
            }
            self.pos += 1;
            self.ws();
            let v = self.parse_value();
            if let Some(slot) = map.iter_mut().find(|(ek, _)| *ek == k) {
                slot.1 = v;
            } else {
                map.push((k, v));
            }
            self.ws();
            match self.peek() {
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                Some(',') => {
                    let comma = self.pos;
                    self.pos += 1;
                    self.ws();
                    if PY_MINOR >= 13 && self.peek() == Some('}') {
                        self.err_at("Illegal trailing comma before end of object", comma);
                    }
                }
                _ => self.err_at("Expecting ',' delimiter", self.pos),
            }
        }
        PyValue::Dict(map)
    }

    fn parse_array(&mut self) -> PyValue {
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return PyValue::List(items);
        }
        loop {
            self.ws();
            items.push(self.parse_value());
            self.ws();
            match self.peek() {
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                Some(',') => {
                    let comma = self.pos;
                    self.pos += 1;
                    self.ws();
                    if PY_MINOR >= 13 && self.peek() == Some(']') {
                        self.err_at("Illegal trailing comma before end of array", comma);
                    }
                }
                _ => self.err_at("Expecting ',' delimiter", self.pos),
            }
        }
        PyValue::List(items)
    }

    fn hex4(&mut self, esc_pos: usize) -> u32 {
        if self.pos + 4 > self.chars.len() {
            self.err_at("Invalid \\uXXXX escape", esc_pos);
        }
        let mut code = 0u32;
        for i in 0..4 {
            match self.chars[self.pos + i].to_digit(16) {
                Some(d) => code = code * 16 + d,
                None => self.err_at("Invalid \\uXXXX escape", esc_pos),
            }
        }
        self.pos += 4;
        code
    }

    fn parse_string(&mut self, start: usize) -> String {
        let mut out = String::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => self.err_at("Unterminated string starting at", start),
            };
            self.pos += 1;
            match c {
                '"' => break,
                '\\' => {
                    let esc_pos = self.pos - 1;
                    let e = match self.peek() {
                        Some(e) => e,
                        None => self.err_at("Unterminated string starting at", start),
                    };
                    self.pos += 1;
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\x08'),
                        'f' => out.push('\x0c'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let mut code = self.hex4(esc_pos);
                            if (0xd800..0xdc00).contains(&code) && self.starts_with("\\u") {
                                let save = self.pos;
                                self.pos += 2;
                                let low = self.hex4(save);
                                if (0xdc00..0xe000).contains(&low) {
                                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                                } else {
                                    self.pos = save;
                                }
                            }
                            match char::from_u32(code) {
                                Some(ch) => out.push(ch),
                                None => py_limit("lone surrogate characters cannot be represented in native strings"),
                            }
                        }
                        other => self.err_at(&format!("Invalid \\escape: {}", py_repr_str(&other.to_string())), esc_pos),
                    }
                }
                c if (c as u32) < 0x20 => self.err_at("Invalid control character at", self.pos - 1),
                c => out.push(c),
            }
        }
        out
    }

    fn parse_number(&mut self) -> Option<PyValue> {
        let start = self.pos;
        let digits = |p: &mut Self| {
            let s = p.pos;
            while p.pos < p.chars.len() && p.chars[p.pos].is_ascii_digit() {
                p.pos += 1;
            }
            p.pos - s
        };
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        match self.peek() {
            Some('0') => self.pos += 1,
            Some(c) if c.is_ascii_digit() => { digits(self); }
            _ => { self.pos = start; return None; }
        }
        let mut is_float = false;
        if self.peek() == Some('.') && self.chars.get(self.pos + 1).map_or(false, |c| c.is_ascii_digit()) {
            is_float = true;
            self.pos += 1;
            digits(self);
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.pos += 1;
            }
            if digits(self) == 0 {
                self.pos = save;
            } else {
                is_float = true;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            Some(PyValue::Float(text.parse::<f64>().unwrap()))
        } else {
            match text.parse::<i64>() {
                Ok(v) => Some(PyValue::Int(v)),
                Err(_) => py_limit("JSON integer does not fit in 64 bits"),
            }
        }
    }
}

pub fn py_json_loads(s: &str) -> PyValue {
    JsonParser::new(s).parse_document()
}

/// Dynamic attribute-style access on PyValue containers.
pub fn py_value_get(v: &PyValue, k: &PyValue, default: PyValue) -> PyValue {
    match (v, k) {
        (PyValue::Dict(map), PyValue::Str(s)) => map.iter().find(|(ek, _)| ek == s).map(|(_, vv)| vv.clone()).unwrap_or(default),
        (PyValue::Dict(_), _) => default,
        (other, _) => py_raise(format!("AttributeError: '{}' object has no attribute 'get'", other.type_name())),
    }
}

pub fn py_value_contains(c: &PyValue, v: &PyValue) -> bool {
    match c {
        PyValue::Dict(map) => match v {
            PyValue::Str(s) => map.iter().any(|(k, _)| k == s),
            _ => false,
        },
        PyValue::List(items) => items.iter().any(|x| x == v),
        PyValue::Str(s) => match v {
            PyValue::Str(sub) => s.contains(sub.as_str()),
            other => py_raise(format!("TypeError: 'in <string>' requires string as left operand, not {}", other.type_name())),
        },
        other => py_raise(format!("TypeError: argument of type '{}' is not iterable", other.type_name())),
    }
}

pub fn py_dyn_index(v: &PyValue, k: &PyValue) -> PyValue {
    match (v, k) {
        (PyValue::Dict(map), _) => match k {
            PyValue::Str(s) => match map.iter().find(|(ek, _)| ek == s) {
                Some((_, vv)) => vv.clone(),
                None => py_raise(format!("KeyError: {}", k.py_repr())),
            },
            _ => py_raise(format!("KeyError: {}", k.py_repr())),
        },
        (PyValue::List(items), PyValue::Int(_) | PyValue::Bool(_)) => py_index(items, k.as_i64()),
        (PyValue::Str(s), PyValue::Int(_) | PyValue::Bool(_)) => PyValue::Str(py_str_index(s, k.as_i64())),
        (PyValue::List(_) | PyValue::Str(_), other) => py_raise(format!(
            "TypeError: {} indices must be integers or slices, not {}",
            v.type_name(),
            other.type_name()
        )),
        (other, _) => py_raise(format!("TypeError: '{}' object is not subscriptable", other.type_name())),
    }
}

pub fn py_value_keys(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::Dict(map) => map.iter().map(|(k, _)| PyValue::Str(k.clone())).collect(),
        other => py_raise(format!("AttributeError: '{}' object has no attribute 'keys'", other.type_name())),
    }
}

pub fn py_value_values(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::Dict(map) => map.iter().map(|(_, x)| x.clone()).collect(),
        other => py_raise(format!("AttributeError: '{}' object has no attribute 'values'", other.type_name())),
    }
}

pub fn py_value_items(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::Dict(map) => map.iter().map(|(k, vv)| PyValue::List(vec![PyValue::Str(k.clone()), vv.clone()])).collect(),
        other => py_raise(format!("AttributeError: '{}' object has no attribute 'items'", other.type_name())),
    }
}

pub fn py_value_iter(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::List(items) => items.clone(),
        PyValue::Dict(map) => map.iter().map(|(k, _)| PyValue::Str(k.clone())).collect(),
        PyValue::Str(s) => s.chars().map(|c| PyValue::Str(c.to_string())).collect(),
        other => py_raise(format!("TypeError: '{}' object is not iterable", other.type_name())),
    }
}

// ---------------------------------------------------------------------------
// statistics (exact, for int data; float data stays on CPython)
// ---------------------------------------------------------------------------

fn py_stat_ints(xs: &[PyValue]) -> Vec<i128> {
    xs.iter()
        .map(|v| match v {
            PyValue::Int(i) => *i as i128,
            PyValue::Bool(b) => *b as i128,
            _ => py_limit("statistics on non-integer data is not supported natively"),
        })
        .collect()
}

/// Fraction -> Python result type (int when exact, else correctly rounded float).
fn py_frac_result(num: i128, den: i128) -> PyValue {
    let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
    if num % den == 0 {
        let q = num / den;
        if q < i64::MIN as i128 || q > i64::MAX as i128 {
            py_limit("statistics result does not fit in 64 bits");
        }
        return PyValue::Int(q as i64);
    }
    if num.unsigned_abs() >= (1u128 << 100) || den.unsigned_abs() >= (1u128 << 64) {
        py_limit("statistics intermediate result is too large");
    }
    PyValue::Float(py_ratio_to_f64(num, den))
}

/// Exact sum of finite floats as a big integer scaled by 2**1074
/// (little-endian u32 limbs, sign-magnitude), for statistics on floats.
struct PyBigFix {
    neg: bool,
    mag: Vec<u32>,
}

impl PyBigFix {
    fn zero() -> Self {
        PyBigFix { neg: false, mag: vec![0; 72] } // 2304 bits: 2**1074 * 2**1024 * n
    }

    fn add_mag(a: &mut Vec<u32>, b: &[u32]) {
        let mut carry = 0u64;
        for i in 0..a.len() {
            let v = a[i] as u64 + *b.get(i).unwrap_or(&0) as u64 + carry;
            a[i] = v as u32;
            carry = v >> 32;
        }
        if carry != 0 {
            a.push(carry as u32);
        }
    }

    /// a -= b, requires a >= b
    fn sub_mag(a: &mut Vec<u32>, b: &[u32]) {
        let mut borrow = 0i64;
        for i in 0..a.len() {
            let mut v = a[i] as i64 - *b.get(i).unwrap_or(&0) as i64 - borrow;
            borrow = 0;
            if v < 0 {
                v += 1 << 32;
                borrow = 1;
            }
            a[i] = v as u32;
        }
    }

    fn cmp_mag(a: &[u32], b: &[u32]) -> Ordering {
        let n = a.len().max(b.len());
        for i in (0..n).rev() {
            let (x, y) = (*a.get(i).unwrap_or(&0), *b.get(i).unwrap_or(&0));
            if x != y {
                return x.cmp(&y);
            }
        }
        Ordering::Equal
    }

    fn add_f64(&mut self, x: f64) {
        if x == 0.0 {
            return;
        }
        let bits = x.to_bits();
        let neg = (bits >> 63) == 1;
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        let (m, e) = if exp == 0 { (frac, -1074i64) } else { (frac | (1u64 << 52), exp - 1075) };
        // x = m * 2**e ; scaled by 2**1074 -> m << (e + 1074)
        let shift = (e + 1074) as usize;
        let mut term = vec![0u32; 72];
        let (w, b) = (shift / 32, shift % 32);
        let wide = (m as u128) << b;
        for k in 0..3 {
            let limb = (wide >> (32 * k)) as u32;
            if w + k < term.len() {
                term[w + k] = limb;
            }
        }
        if neg == self.neg {
            Self::add_mag(&mut self.mag, &term);
        } else if Self::cmp_mag(&self.mag, &term) != Ordering::Less {
            Self::sub_mag(&mut self.mag, &term);
        } else {
            let mut t = term;
            Self::sub_mag(&mut t, &self.mag);
            self.mag = t;
            self.neg = neg;
        }
    }

    /// Correctly rounded value of self / (n * 2**1074).
    fn div_to_f64(&self, n: u64) -> f64 {
        // Long division by n, keeping the remainder as a sticky bit.
        let mut q = vec![0u32; self.mag.len()];
        let mut rem: u64 = 0;
        for i in (0..self.mag.len()).rev() {
            let cur = (rem << 32) | self.mag[i] as u64;
            q[i] = (cur / n) as u32;
            rem = cur % n;
        }
        let mut sticky = rem != 0;
        let neg_zero = |v: f64| if self.neg { -v } else { v };
        let top = match q.iter().rposition(|&l| l != 0) {
            Some(t) => t,
            None => {
                // Below one unit of 2**-1074: round rem/n to 0 or 1 ulp, half-even.
                let twice = 2 * rem as u128;
                return neg_zero(if twice > n as u128 { 5e-324 } else { 0.0 });
            }
        };
        let bitlen = top * 32 + (32 - q[top].leading_zeros() as usize); // value = q * 2**-1074
        // Keep 53 bits (fewer for subnormals): the unit in the last place is 2**lsb.
        let exp_of_msb = bitlen as i64 - 1 - 1074;
        let lsb = (exp_of_msb - 52).max(-1074);
        let cut = (lsb + 1074) as usize; // number of low bits of q to drop
        let get_bit = |i: usize| -> u32 { if i / 32 < q.len() { (q[i / 32] >> (i % 32)) & 1 } else { 0 } };
        let mut mant: u64 = 0;
        for i in (cut..bitlen).rev() {
            mant = (mant << 1) | get_bit(i) as u64;
        }
        let (half, from_rem) = if cut > 0 {
            (get_bit(cut - 1), false)
        } else {
            // The ulp is 2**-1074 itself: the remainder is the dropped fraction.
            let twice = 2 * rem as u128;
            (if twice >= n as u128 { 1 } else { 0 }, true)
        };
        if from_rem {
            sticky = (2 * rem as u128) != n as u128 && rem != 0;
        }
        if cut > 1 {
            for i in 0..cut - 1 {
                if get_bit(i) == 1 {
                    sticky = true;
                    break;
                }
            }
        }
        if half == 1 && (sticky || (mant & 1) == 1) {
            mant += 1;
        }
        // Scale in two exact steps: 2**lsb alone may underflow (2**-1049).
        let (h1, h2) = ((lsb / 2) as i32, (lsb - lsb / 2) as i32);
        let v = (mant as f64) * 2f64.powi(h1) * 2f64.powi(h2);
        let v = if v.is_finite() { v } else { f64::INFINITY };
        if self.neg { -v } else { v }
    }
}

/// statistics.mean of float data (CPython: exact fraction sum, rounded once).
pub fn py_mean_floats(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        py_raise("StatisticsError: mean requires at least one data point".to_string());
    }
    if xs.iter().any(|x| !x.is_finite()) {
        py_unsupported("statistics.mean of data containing inf or nan");
    }
    let mut acc = PyBigFix::zero();
    for &x in xs {
        acc.add_f64(x);
    }
    acc.div_to_f64(xs.len() as u64)
}

pub fn py_mean(xs: &[PyValue]) -> PyValue {
    if xs.is_empty() {
        py_raise("StatisticsError: mean requires at least one data point".to_string());
    }
    let v = py_stat_ints(xs);
    let s: i128 = v.iter().sum();
    py_frac_result(s, v.len() as i128)
}

fn py_ss_frac(v: &[i128]) -> (i128, i128) {
    // sum((x - mean)^2) = (n*sum(x^2) - sum(x)^2) / n
    let n = v.len() as i128;
    let s: i128 = v.iter().sum();
    let s2 = v.iter().try_fold(0i128, |acc, x| x.checked_mul(*x).and_then(|sq| acc.checked_add(sq)));
    let s2 = match s2 { Some(x) => x, None => py_limit("statistics intermediate result is too large") };
    let num = n.checked_mul(s2).and_then(|a| s.checked_mul(s).and_then(|b| a.checked_sub(b)));
    match num {
        Some(x) => (x, n),
        None => py_limit("statistics intermediate result is too large"),
    }
}

pub fn py_variance(xs: &[PyValue]) -> PyValue {
    if xs.len() < 2 {
        py_raise("StatisticsError: variance requires at least two data points".to_string());
    }
    let v = py_stat_ints(xs);
    let (num, n) = py_ss_frac(&v);
    py_frac_result(num, n * (n - 1))
}

pub fn py_pvariance(xs: &[PyValue]) -> PyValue {
    if xs.is_empty() {
        py_raise("StatisticsError: pvariance requires at least one data point".to_string());
    }
    let v = py_stat_ints(xs);
    let (num, n) = py_ss_frac(&v);
    py_frac_result(num, n * n)
}

fn py_sorted_values(xs: &[PyValue]) -> Vec<PyValue> {
    let mut v = xs.to_vec();
    py_sort(&mut v);
    v
}

pub fn py_median(xs: &[PyValue]) -> PyValue {
    if xs.is_empty() {
        py_raise("StatisticsError: no median for empty data".to_string());
    }
    let v = py_sorted_values(xs);
    let n = v.len();
    if n % 2 == 1 {
        return v[n / 2].clone();
    }
    (v[n / 2 - 1].clone() + v[n / 2].clone()) / PyValue::Int(2)
}

pub fn py_median_low(xs: &[PyValue]) -> PyValue {
    if xs.is_empty() {
        py_raise("StatisticsError: no median for empty data".to_string());
    }
    let v = py_sorted_values(xs);
    let n = v.len();
    if n % 2 == 1 { v[n / 2].clone() } else { v[n / 2 - 1].clone() }
}

pub fn py_median_high(xs: &[PyValue]) -> PyValue {
    if xs.is_empty() {
        py_raise("StatisticsError: no median for empty data".to_string());
    }
    let v = py_sorted_values(xs);
    v[v.len() / 2].clone()
}

pub fn py_mode<T: Clone + PartialEq>(xs: &[T]) -> T {
    if xs.is_empty() {
        py_raise("StatisticsError: no mode for empty data".to_string());
    }
    let mut best = 0usize;
    let mut best_n = 0usize;
    for (i, v) in xs.iter().enumerate() {
        if xs[..i].iter().any(|w| w == v) {
            continue;
        }
        let n = xs.iter().filter(|w| *w == v).count();
        if n > best_n {
            best_n = n;
            best = i;
        }
    }
    xs[best].clone()
}

pub fn py_gc_collect() -> i64 { 0 }
pub fn py_gc_isenabled() -> bool { true }
pub fn py_gc_disable() {}
pub fn py_gc_enable() {}
pub fn py_gc_get_count() -> (i64, i64, i64) { (0, 0, 0) }

pub fn py_format_commas(n: i64) -> String {
    let mut digits: Vec<u8> = n.unsigned_abs().to_string().into_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(digits.len() + digits.len() / 3);
    let mut count = 0;
    while let Some(ch) = digits.pop() {
        if count == 3 {
            out.push(b',');
            count = 0;
        }
        out.push(ch);
        count += 1;
    }
    if n < 0 {
        out.push(b'-');
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Files: shared handles (like Python file objects), opened eagerly
// ---------------------------------------------------------------------------

pub fn py_os_error_text(e: &io::Error) -> String {
    let text = e.to_string();
    let text = match text.find(" (os error") {
        Some(i) => text[..i].to_string(),
        None => text,
    };
    match e.raw_os_error() {
        Some(code) => format!("[Errno {}] {}", code, text),
        None => text,
    }
}

fn py_os_error(e: &io::Error, path: &str) -> ! {
    let kind = match e.kind() {
        io::ErrorKind::NotFound => "FileNotFoundError",
        io::ErrorKind::PermissionDenied => "PermissionError",
        io::ErrorKind::AlreadyExists => "FileExistsError",
        _ => match e.raw_os_error() {
            Some(21) => "IsADirectoryError",
            Some(20) => "NotADirectoryError",
            _ => "OSError",
        },
    };
    py_raise(format!("{}: {}: {}", kind, py_os_error_text(e), py_repr_str(path)))
}

struct PyFileInner {
    path: String,
    mode: String,
    text: Vec<char>,
    pos: usize,
    writer: Option<io::BufWriter<std::fs::File>>,
    closed: bool,
}

impl PyFileInner {
    fn flush(&mut self) {
        if let Some(w) = self.writer.as_mut() {
            if let Err(e) = w.flush() {
                let p = self.path.clone();
                py_os_error(&e, &p);
            }
        }
    }
}

impl Drop for PyFileInner {
    fn drop(&mut self) {
        if let Some(w) = self.writer.as_mut() {
            let _ = w.flush();
        }
    }
}

thread_local! {
    static PY_OPEN_FILES: RefCell<Vec<Weak<RefCell<PyFileInner>>>> = const { RefCell::new(Vec::new()) };
}

pub fn py_flush_files() {
    let files: Vec<Rc<RefCell<PyFileInner>>> =
        PY_OPEN_FILES.with(|f| f.borrow().iter().filter_map(|w| w.upgrade()).collect());
    for f in files {
        if let Ok(mut inner) = f.try_borrow_mut() {
            if let Some(w) = inner.writer.as_mut() {
                let _ = w.flush();
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct PyFile {
    inner: Option<Rc<RefCell<PyFileInner>>>,
}

impl fmt::Debug for PyFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PyFile")
    }
}

impl PartialEq for PyFile {
    fn eq(&self, other: &Self) -> bool {
        match (&self.inner, &other.inner) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
}

fn py_decode_text(bytes: Vec<u8>, path: &str) -> Vec<char> {
    let s = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            let pos = e.utf8_error().valid_up_to();
            let b = e.as_bytes()[pos];
            py_raise(format!(
                "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x{:02x} in position {}: invalid start byte",
                b, pos
            ))
        }
    };
    // Universal newlines.
    let mut out = Vec::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    out
}

impl PyFile {
    fn new_inner(inner: PyFileInner) -> Self {
        let rc = Rc::new(RefCell::new(inner));
        PY_OPEN_FILES.with(|f| {
            let mut v = f.borrow_mut();
            v.retain(|w| w.strong_count() > 0);
            v.push(Rc::downgrade(&rc));
        });
        PyFile { inner: Some(rc) }
    }

    pub fn open_read(path: &str) -> Self {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                if std::path::Path::new(path).is_dir() {
                    py_raise(format!("IsADirectoryError: [Errno 21] Is a directory: {}", py_repr_str(path)));
                }
                py_os_error(&e, path)
            }
        };
        let text = py_decode_text(bytes, path);
        Self::new_inner(PyFileInner { path: path.to_string(), mode: "r".into(), text, pos: 0, writer: None, closed: false })
    }

    fn open_writable(path: &str, append: bool) -> Self {
        py_mark_effect();
        let mut opts = std::fs::OpenOptions::new();
        if append {
            opts.append(true).create(true);
        } else {
            opts.write(true).create(true).truncate(true);
        }
        let file = match opts.open(path) {
            Ok(f) => f,
            Err(e) => {
                if std::path::Path::new(path).is_dir() {
                    py_raise(format!("IsADirectoryError: [Errno 21] Is a directory: {}", py_repr_str(path)));
                }
                py_os_error(&e, path)
            }
        };
        Self::new_inner(PyFileInner {
            path: path.to_string(),
            mode: if append { "a".into() } else { "w".into() },
            text: Vec::new(),
            pos: 0,
            writer: Some(io::BufWriter::new(file)),
            closed: false,
        })
    }

    pub fn open_write(path: &str) -> Self {
        Self::open_writable(path, false)
    }

    pub fn open_append(path: &str) -> Self {
        Self::open_writable(path, true)
    }

    fn with<R>(&self, f: impl FnOnce(&mut PyFileInner) -> R) -> R {
        match &self.inner {
            Some(rc) => {
                let mut inner = rc.borrow_mut();
                if inner.closed {
                    py_raise("ValueError: I/O operation on closed file.".to_string());
                }
                f(&mut inner)
            }
            None => py_raise("ValueError: I/O operation on closed file.".to_string()),
        }
    }

    pub fn read(&self) -> String {
        self.with(|f| {
            if f.writer.is_some() {
                py_raise("io.UnsupportedOperation: not readable".to_string());
            }
            let s: String = f.text[f.pos..].iter().collect();
            f.pos = f.text.len();
            s
        })
    }

    pub fn readline(&self) -> String {
        self.with(|f| {
            if f.writer.is_some() {
                py_raise("io.UnsupportedOperation: not readable".to_string());
            }
            let start = f.pos;
            while f.pos < f.text.len() {
                f.pos += 1;
                if f.text[f.pos - 1] == '\n' {
                    break;
                }
            }
            f.text[start..f.pos].iter().collect()
        })
    }

    pub fn readlines(&self) -> Vec<String> {
        let mut out = Vec::new();
        loop {
            let line = self.readline();
            if line.is_empty() {
                break;
            }
            out.push(line);
        }
        out
    }

    pub fn lines(&self) -> Vec<String> {
        self.readlines()
    }

    pub fn write(&mut self, s: &str) -> i64 {
        self.with(|f| {
            let path = f.path.clone();
            match f.writer.as_mut() {
                Some(w) => {
                    let data = if cfg!(windows) { s.replace('\n', "\r\n") } else { s.to_string() };
                    if let Err(e) = w.write_all(data.as_bytes()) {
                        py_os_error(&e, &path);
                    }
                }
                None => py_raise("io.UnsupportedOperation: not writable".to_string()),
            }
            s.chars().count() as i64
        })
    }

    pub fn close(&mut self) {
        if let Some(rc) = &self.inner {
            let mut inner = rc.borrow_mut();
            if inner.closed {
                return;
            }
            inner.flush();
            inner.writer = None;
            inner.closed = true;
        }
    }
}

// ---------------------------------------------------------------------------
// dict: insertion-ordered, hashed (Python 3.7+ semantics)
// ---------------------------------------------------------------------------

/// Python-hashable key types. Keys that compare equal must hash equal.
pub trait PyKey: Clone + PartialEq {
    fn py_hash(&self) -> u64;
}

#[inline]
fn py_mix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

impl PyKey for i64 {
    fn py_hash(&self) -> u64 { py_mix(*self as u64) }
}
impl PyKey for bool {
    fn py_hash(&self) -> u64 { py_mix(*self as u64) }
}
impl PyKey for f64 {
    fn py_hash(&self) -> u64 {
        if self.fract() == 0.0 && self.abs() < 9.2e18 {
            py_mix(*self as i64 as u64)
        } else {
            py_mix(self.to_bits())
        }
    }
}
impl PyKey for () {
    fn py_hash(&self) -> u64 { 0x5bd1e995 }
}
impl PyKey for String {
    fn py_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.as_str().hash(&mut h);
        h.finish()
    }
}
impl PyKey for PyValue {
    fn py_hash(&self) -> u64 {
        match self {
            PyValue::None => ().py_hash(),
            PyValue::Bool(b) => (*b as i64).py_hash(),
            PyValue::Int(i) => i.py_hash(),
            PyValue::Float(f) => f.py_hash(),
            PyValue::Str(s) => s.py_hash(),
            other => py_raise(format!("TypeError: unhashable type: '{}'", other.type_name())),
        }
    }
}

macro_rules! py_key_tuple {
    ($($name:ident : $idx:tt),+) => {
        impl<$($name: PyKey),+> PyKey for ($($name,)+) {
            fn py_hash(&self) -> u64 {
                let mut h: u64 = 0x345678;
                $( h = py_mix(h ^ self.$idx.py_hash()).wrapping_add($idx); )+
                h
            }
        }
    };
}
py_key_tuple!(A: 0);
py_key_tuple!(A: 0, B: 1);
py_key_tuple!(A: 0, B: 1, C: 2);
py_key_tuple!(A: 0, B: 1, C: 2, D: 3);
py_key_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4);
py_key_tuple!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);

#[derive(Default, Clone, Copy)]
pub struct PyIdHasher(u64);

impl Hasher for PyIdHasher {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 << 8) ^ (*b as u64);
        }
    }
    fn write_u64(&mut self, n: u64) { self.0 = n; }
}

type PyIndex = HashMap<u64, Vec<u32>, BuildHasherDefault<PyIdHasher>>;

#[derive(Clone)]
pub struct PyDict<K, V> {
    entries: Vec<Option<(K, V)>>,
    index: PyIndex,
    live: usize,
}

impl<K, V> Default for PyDict<K, V> {
    fn default() -> Self {
        PyDict { entries: Vec::new(), index: PyIndex::default(), live: 0 }
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for PyDict<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.entries.iter().flatten().map(|(k, v)| (k, v))).finish()
    }
}

fn py_key_error<K: PyRepr>(k: &K) -> ! {
    py_raise(format!("KeyError: {}", k.py_repr()))
}

impl<K: PyKey + PyRepr, V> std::ops::Index<&K> for PyDict<K, V> {
    type Output = V;
    fn index(&self, k: &K) -> &V {
        match self.get(k) {
            Some(v) => v,
            None => py_key_error(k),
        }
    }
}

impl<K: PyKey, V> FromIterator<(K, V)> for PyDict<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = PyDict::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: PyKey, V> PyDict<K, V> {
    pub fn new() -> Self {
        PyDict::default()
    }

    pub fn from_vec(entries: Vec<(K, V)>) -> Self {
        entries.into_iter().collect()
    }

    fn find(&self, k: &K) -> Option<usize> {
        let slots = self.index.get(&k.py_hash())?;
        for &i in slots {
            if let Some((ek, _)) = &self.entries[i as usize] {
                if ek == k {
                    return Some(i as usize);
                }
            }
        }
        None
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        match self.find(&k) {
            Some(i) => {
                let slot = self.entries[i].as_mut().unwrap();
                Some(std::mem::replace(&mut slot.1, v))
            }
            None => {
                let h = k.py_hash();
                self.index.entry(h).or_default().push(self.entries.len() as u32);
                self.entries.push(Some((k, v)));
                self.live += 1;
                None
            }
        }
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.find(k).map(|i| &self.entries[i].as_ref().unwrap().1)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        match self.find(k) {
            Some(i) => Some(&mut self.entries[i].as_mut().unwrap().1),
            None => None,
        }
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        let i = self.find(k)?;
        let h = k.py_hash();
        if let Some(slots) = self.index.get_mut(&h) {
            slots.retain(|&s| s as usize != i);
            if slots.is_empty() {
                self.index.remove(&h);
            }
        }
        let (_, v) = self.entries[i].take().unwrap();
        self.live -= 1;
        if self.entries.len() > 16 && self.live * 2 < self.entries.len() {
            self.compact();
        }
        Some(v)
    }

    fn compact(&mut self) {
        let old = std::mem::take(&mut self.entries);
        self.index.clear();
        self.live = 0;
        for (k, v) in old.into_iter().flatten() {
            let h = k.py_hash();
            self.index.entry(h).or_default().push(self.entries.len() as u32);
            self.entries.push(Some((k, v)));
            self.live += 1;
        }
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.find(k).is_some()
    }

    pub fn entry(&mut self, k: K) -> PyDictEntry<'_, K, V> {
        PyDictEntry { map: self, key: k }
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.live = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter().flatten()
    }

    /// `for v in d.values()` modifying each value in place.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.entries.iter_mut().flatten().map(|(_, v)| v)
    }

    /// `for k, v in d.items()` modifying each value in place.
    pub fn items_mut(&mut self) -> impl Iterator<Item = (K, &mut V)>
    where
        K: Clone,
    {
        self.entries.iter_mut().flatten().map(|(k, v)| (k.clone(), v))
    }

    pub fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }

    pub fn into_vec(self) -> Vec<(K, V)> {
        let mut me = self;
        std::mem::take(&mut me.entries).into_iter().flatten().collect()
    }
}

impl<K: PyKey + PyRepr, V> PyDict<K, V> {
    /// `d[k]` as a mutable place, raising KeyError like Python.
    pub fn index_mut(&mut self, k: &K) -> &mut V {
        match self.find(k) {
            Some(i) => &mut self.entries[i].as_mut().unwrap().1,
            None => py_key_error(k),
        }
    }

    pub fn pop_key(&mut self, k: &K) -> V {
        match self.remove(k) {
            Some(v) => v,
            None => py_key_error(k),
        }
    }

    pub fn del_key(&mut self, k: &K) {
        if self.remove(k).is_none() {
            py_key_error(k);
        }
    }
}

impl<K: PyKey, V: Clone> PyDict<K, V> {
    pub fn keys(&self) -> Vec<K> {
        self.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn values(&self) -> Vec<V> {
        self.iter().map(|(_, v)| v.clone()).collect()
    }

    pub fn items(&self) -> Vec<(K, V)> {
        self.iter().cloned().collect()
    }
}

impl<K: PyKey, V: PartialEq> PartialEq for PyDict<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

pub struct PyDictEntry<'a, K, V> {
    map: &'a mut PyDict<K, V>,
    key: K,
}

impl<'a, K: PyKey, V> PyDictEntry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        let i = match self.map.find(&self.key) {
            Some(i) => i,
            None => {
                self.map.insert(self.key, default);
                self.map.entries.len() - 1
            }
        };
        &mut self.map.entries[i].as_mut().unwrap().1
    }
}

// ---------------------------------------------------------------------------
// time
// ---------------------------------------------------------------------------

pub fn py_os_getenv(key: &str) -> PyValue {
    match std::env::var(key) {
        Ok(v) => PyValue::Str(v),
        Err(_) => PyValue::None,
    }
}

pub fn py_os_getcwd() -> String {
    match std::env::current_dir() {
        Ok(p) => p.display().to_string(),
        Err(e) => py_raise(format!("OSError: {}", py_os_error_text(&e))),
    }
}

pub fn py_os_listdir(path: &str) -> Vec<String> {
    match std::fs::read_dir(path) {
        Ok(rd) => rd.filter_map(|e| e.ok().and_then(|d| d.file_name().into_string().ok())).collect(),
        Err(e) => py_os_error(&e, path),
    }
}

pub fn py_os_mkdir(path: &str, recursive: bool, exist_ok: bool) {
    py_mark_effect();
    let p = std::path::Path::new(path);
    if p.exists() {
        if exist_ok && p.is_dir() {
            return;
        }
        py_raise(format!("FileExistsError: [Errno 17] File exists: {}", py_repr_str(path)));
    }
    let r = if recursive { std::fs::create_dir_all(p) } else { std::fs::create_dir(p) };
    if let Err(e) = r {
        py_os_error(&e, path);
    }
}

pub fn py_os_remove(path: &str) {
    py_mark_effect();
    // Same syscall as CPython (unlink), so the same errno and message.
    if let Err(e) = std::fs::remove_file(path) {
        py_os_error(&e, path);
    }
}

pub fn py_unix_time() -> f64 {
    py_mark_nondet();
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

static PY_T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

pub fn py_perf_counter() -> f64 {
    py_mark_nondet();
    PY_T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

pub fn py_sleep(secs: f64) {
    if secs < 0.0 {
        py_raise("ValueError: sleep length must be non-negative".to_string());
    }
    let _ = py_flush_stdout();
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
}
"""


def _build() -> str:
    from pyrust.unicode_tables import rust_tables

    return _TEMPLATE.replace("@@PY_MINOR@@", str(sys.version_info[1])).replace(
        "@@UNICODE_TABLES@@", rust_tables()
    )


RUNTIME_RS = _build()
