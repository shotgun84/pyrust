from __future__ import annotations

RUNTIME_RS = r"""#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::io::{self, Write};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

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

impl PartialEq for PyValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PyValue::None, PyValue::None) => true,
            (PyValue::Bool(a), PyValue::Bool(b)) => a == b,
            (PyValue::Int(a), PyValue::Int(b)) => a == b,
            (PyValue::Float(a), PyValue::Float(b)) => a == b,
            (PyValue::Str(a), PyValue::Str(b)) => a == b,
            (PyValue::List(a), PyValue::List(b)) => a == b,
            (PyValue::Dict(a), PyValue::Dict(b)) => {
                // Order-insensitive, like Python dict ==.
                a.len() == b.len()
                    && a.iter().all(|(k, v)| b.iter().any(|(ok, ov)| ok == k && ov == v))
            }
            _ => false,
        }
    }
}

impl Default for PyValue {
    fn default() -> Self {
        PyValue::None
    }
}

impl PartialOrd for PyValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (PyValue::Int(a), PyValue::Int(b)) => a.partial_cmp(b),
            (PyValue::Str(a), PyValue::Str(b)) => a.partial_cmp(b),
            (a, b) => a.as_f64().partial_cmp(&b.as_f64()),
        }
    }
}

impl PyValue {
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

    pub fn as_f64(&self) -> f64 {
        match self {
            PyValue::Int(i) => *i as f64,
            PyValue::Float(f) => *f,
            PyValue::Bool(b) => if *b { 1.0 } else { 0.0 },
            PyValue::Str(s) => s.parse::<f64>().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            PyValue::Int(i) => *i,
            PyValue::Float(f) => *f as i64,
            PyValue::Bool(b) => if *b { 1 } else { 0 },
            PyValue::Str(s) => s.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            PyValue::Str(s) => s.chars().count(),
            PyValue::List(v) => v.len(),
            PyValue::Dict(m) => m.len(),
            _ => 0,
        }
    }
}

impl fmt::Display for PyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyValue::None => write!(f, "None"),
            PyValue::Bool(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            PyValue::Int(i) => write!(f, "{}", i),
            PyValue::Float(fl) => {
                if fl.fract() == 0.0 {
                    write!(f, "{:.1}", fl)
                } else {
                    write!(f, "{}", fl)
                }
            }
            PyValue::Str(s) => write!(f, "{}", s),
            PyValue::List(items) => {
                let parts: Vec<String> = items.iter().map(py_value_repr).collect();
                write!(f, "[{}]", parts.join(", "))
            }
            PyValue::Dict(map) => {
                let parts: Vec<String> = map.iter().map(|(k, v)| format!("{}: {}", py_repr_str(k), py_value_repr(v))).collect();
                write!(f, "{{{}}}", parts.join(", "))
            }
        }
    }
}

impl From<()> for PyValue {
    fn from(_: ()) -> Self {
        PyValue::None
    }
}

impl From<i64> for PyValue {
    fn from(v: i64) -> Self {
        PyValue::Int(v)
    }
}

impl From<f64> for PyValue {
    fn from(v: f64) -> Self {
        PyValue::Float(v)
    }
}

impl From<bool> for PyValue {
    fn from(v: bool) -> Self {
        PyValue::Bool(v)
    }
}

impl From<String> for PyValue {
    fn from(v: String) -> Self {
        PyValue::Str(v)
    }
}

impl From<&str> for PyValue {
    fn from(v: &str) -> Self {
        PyValue::Str(v.to_string())
    }
}

impl<T: Into<PyValue> + Clone> From<Vec<T>> for PyValue {
    fn from(v: Vec<T>) -> Self {
        PyValue::List(v.into_iter().map(|x| x.into()).collect())
    }
}

impl<V: Into<PyValue> + Clone> From<PyDict<String, V>> for PyValue {
    fn from(m: PyDict<String, V>) -> Self {
        PyValue::Dict(m.entries.into_iter().map(|(k, v)| (k, v.into())).collect())
    }
}

pub fn py_repr_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\'' => out.push_str("\\'"),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

pub fn py_value_repr(v: &PyValue) -> String {
    match v {
        PyValue::Str(s) => py_repr_str(s),
        PyValue::List(items) => {
            let parts: Vec<String> = items.iter().map(py_value_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        PyValue::Dict(map) => {
            let parts: Vec<String> = map.iter().map(|(k, vv)| format!("{}: {}", py_repr_str(k), py_value_repr(vv))).collect();
            format!("{{{}}}", parts.join(", "))
        }
        other => format!("{}", other),
    }
}

impl Add for PyValue {
    type Output = PyValue;
    fn add(self, rhs: PyValue) -> PyValue {
        match (self, rhs) {
            (PyValue::Int(a), PyValue::Int(b)) => PyValue::Int(a + b),
            (PyValue::Float(a), PyValue::Float(b)) => PyValue::Float(a + b),
            (PyValue::Int(a), PyValue::Float(b)) => PyValue::Float(a as f64 + b),
            (PyValue::Float(a), PyValue::Int(b)) => PyValue::Float(a + b as f64),
            (PyValue::Str(a), PyValue::Str(b)) => PyValue::Str(format!("{}{}", a, b)),
            (PyValue::List(mut a), PyValue::List(b)) => {
                a.extend(b);
                PyValue::List(a)
            }
            (a, b) => PyValue::Float(a.as_f64() + b.as_f64()),
        }
    }
}

impl Sub for PyValue {
    type Output = PyValue;
    fn sub(self, rhs: PyValue) -> PyValue {
        match (self, rhs) {
            (PyValue::Int(a), PyValue::Int(b)) => PyValue::Int(a - b),
            (a, b) => PyValue::Float(a.as_f64() - b.as_f64()),
        }
    }
}

impl Mul for PyValue {
    type Output = PyValue;
    fn mul(self, rhs: PyValue) -> PyValue {
        match (self, rhs) {
            (PyValue::Int(a), PyValue::Int(b)) => PyValue::Int(a * b),
            (PyValue::Str(s), PyValue::Int(n)) | (PyValue::Int(n), PyValue::Str(s)) => {
                PyValue::Str(s.repeat(n.max(0) as usize))
            }
            (a, b) => PyValue::Float(a.as_f64() * b.as_f64()),
        }
    }
}

impl Div for PyValue {
    type Output = PyValue;
    fn div(self, rhs: PyValue) -> PyValue {
        PyValue::Float(py_div_f64(self.as_f64(), rhs.as_f64()))
    }
}

impl Rem for PyValue {
    type Output = PyValue;
    fn rem(self, rhs: PyValue) -> PyValue {
        match (self, rhs) {
            (PyValue::Int(a), PyValue::Int(b)) => PyValue::Int(py_mod(a, b)),
            (a, b) => PyValue::Float(a.as_f64() % b.as_f64()),
        }
    }
}

impl Neg for PyValue {
    type Output = PyValue;
    fn neg(self) -> PyValue {
        match self {
            PyValue::Int(i) => PyValue::Int(-i),
            PyValue::Float(f) => PyValue::Float(-f),
            other => PyValue::Float(-other.as_f64()),
        }
    }
}

#[inline]
pub fn py_floordiv(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("ZeroDivisionError: integer division or modulo by zero");
    }
    if b == -1 {
        return a.checked_neg().expect("OverflowError: integer result exceeds PyRust's 64-bit int range");
    }
    let d = a / b;
    let r = a % b;
    if (r > 0 && b < 0) || (r < 0 && b > 0) {
        d - 1
    } else {
        d
    }
}

#[inline]
pub fn py_floordiv_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        panic!("ZeroDivisionError: float floor division by zero");
    }
    (a / b).floor()
}

#[inline]
pub fn py_val_floordiv(a: PyValue, b: PyValue) -> PyValue {
    match (a, b) {
        (PyValue::Int(x), PyValue::Int(y)) => PyValue::Int(py_floordiv(x, y)),
        (x, y) => PyValue::Float(py_floordiv_f64(x.as_f64(), y.as_f64())),
    }
}

#[inline]
pub fn py_mod(a: i64, b: i64) -> i64 {
    if b == 0 {
        panic!("ZeroDivisionError: integer division or modulo by zero");
    }
    if b == -1 {
        return 0;
    }
    // Sign-corrected remainder; `((a % b) + b) % b` could overflow i64.
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) { r + b } else { r }
}

#[inline]
pub fn py_pow(base: i64, exp: i64) -> i64 {
    // Python promotes to bigint / float here; a 64-bit result that would be
    // wrong must abort loudly instead of printing garbage.
    if exp < 0 {
        panic!("ValueError: negative integer exponent is not supported by PyRust (use a float base)");
    }
    u32::try_from(exp)
        .ok()
        .and_then(|e| base.checked_pow(e))
        .expect("OverflowError: integer result exceeds PyRust's 64-bit int range")
}

/// Python `<<` never loses bits; refuse results that do not fit in i64.
#[inline]
pub fn py_lshift(a: i64, b: i64) -> i64 {
    if b < 0 {
        panic!("ValueError: negative shift count");
    }
    if a == 0 {
        return 0;
    }
    if b >= 63 || ((a << b) >> b) != a {
        panic!("OverflowError: integer result exceeds PyRust's 64-bit int range");
    }
    a << b
}

#[inline]
pub fn py_rshift(a: i64, b: i64) -> i64 {
    if b < 0 {
        panic!("ValueError: negative shift count");
    }
    if b >= 63 { if a < 0 { -1 } else { 0 } } else { a >> b }
}

#[inline]
pub fn py_pow_f64(base: f64, exp: f64) -> f64 {
    if base == 0.0 && exp < 0.0 {
        panic!("ZeroDivisionError: 0.0 cannot be raised to a negative power");
    }
    if base < 0.0 && exp.is_finite() && exp.fract() != 0.0 {
        panic!("ValueError: negative float to a fractional power is complex in Python (unsupported by PyRust)");
    }
    let r = base.powf(exp);
    if r.is_infinite() && base.is_finite() && exp.is_finite() {
        panic!("OverflowError: (34, 'Result too large')");
    }
    r
}

/// Python `/`: true division that raises on a zero divisor instead of
/// silently producing inf/NaN.
#[inline]
pub fn py_div_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        panic!("ZeroDivisionError: float division by zero");
    }
    a / b
}

/// Python `int / int`. CPython rounds the exact quotient correctly even for
/// ints beyond 2**53; casting to f64 first would not, so refuse those.
#[inline]
pub fn py_truediv_i64(a: i64, b: i64) -> f64 {
    const EXACT: i64 = 1 << 53;
    if b == 0 {
        panic!("ZeroDivisionError: division by zero");
    }
    if a > EXACT || a < -EXACT || b > EXACT || b < -EXACT {
        panic!("OverflowError: int / int beyond 2**53 is not exactly representable in PyRust");
    }
    (a as f64) / (b as f64)
}

#[cold]
fn py_math_domain() -> ! {
    panic!("ValueError: math domain error")
}

#[cold]
fn py_math_range() -> ! {
    panic!("OverflowError: math range error")
}

/// `math.*` with CPython's error behaviour (ValueError / OverflowError
/// instead of NaN / inf).
pub fn py_math1(name: &str, x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    match name {
        "sqrt" => { if x < 0.0 { py_math_domain() } x.sqrt() }
        "log" | "ln" => { if x <= 0.0 { py_math_domain() } x.ln() }
        "log10" => { if x <= 0.0 { py_math_domain() } x.log10() }
        "log2" => { if x <= 0.0 { py_math_domain() } x.log2() }
        "sin" => { if x.is_infinite() { py_math_domain() } x.sin() }
        "cos" => { if x.is_infinite() { py_math_domain() } x.cos() }
        "tan" => { if x.is_infinite() { py_math_domain() } x.tan() }
        "asin" => { if x.abs() > 1.0 { py_math_domain() } x.asin() }
        "acos" => { if x.abs() > 1.0 { py_math_domain() } x.acos() }
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

/// `math.floor` / `math.ceil` results: Python returns an exact int (and raises
/// on inf/NaN); refuse values outside the i64 range instead of saturating.
pub fn py_float_to_int(x: f64) -> i64 {
    if !(x > -9.2e18 && x < 9.2e18) {
        panic!("OverflowError: float is too large (or not finite) for PyRust's 64-bit int");
    }
    x as i64
}

#[inline]
pub fn py_val_pow(base: PyValue, exp: PyValue) -> PyValue {
    match (base, exp) {
        (PyValue::Int(x), PyValue::Int(y)) if y >= 0 => PyValue::Int(py_pow(x, y)),
        (x, y) => PyValue::Float(py_pow_f64(x.as_f64(), y.as_f64())),
    }
}

#[inline]
pub fn py_index<T: Clone>(items: &[T], idx: i64) -> T {
    let len = items.len() as i64;
    let resolved = if idx < 0 { len + idx } else { idx };
    items[resolved as usize].clone()
}

#[inline]
pub fn py_set_index<T>(items: &mut [T], idx: i64, val: T) {
    let len = items.len() as i64;
    let resolved = if idx < 0 { len + idx } else { idx };
    items[resolved as usize] = val;
}

pub fn py_slice<T: Clone>(items: &[T], start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<T> {
    let len = items.len() as i64;
    let norm = |bound: i64| -> i64 {
        let b = if bound < 0 { len + bound } else { bound };
        b.clamp(0, len)
    };
    let mut out = Vec::new();
    if step > 0 {
        let mut i = norm(start.unwrap_or(0));
        let end = norm(stop.unwrap_or(len));
        while i < end {
            out.push(items[i as usize].clone());
            i += step;
        }
    } else if step < 0 {
        let mut i = start.map(|s| if s < 0 { len + s } else { s }).unwrap_or(len - 1).min(len - 1);
        let end = stop.map(|s| if s < 0 { len + s } else { s }).unwrap_or(-1);
        while i > end && i >= 0 {
            out.push(items[i as usize].clone());
            i += step;
        }
    }
    out
}

pub fn py_str_index(s: &str, idx: i64) -> String {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len() as i64;
    let resolved = if idx < 0 { len + idx } else { idx };
    chars[resolved as usize].to_string()
}

pub fn py_str_slice(s: &str, start: Option<i64>, stop: Option<i64>, step: i64) -> String {
    let chars: Vec<char> = s.chars().collect();
    py_slice(&chars, start, stop, step).into_iter().collect()
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
    let mut out = Vec::new();
    if step > 0 {
        let mut cur = start;
        while cur < stop {
            out.push(cur);
            cur += step;
        }
    } else if step < 0 {
        let mut cur = start;
        while cur > stop {
            out.push(cur);
            cur += step;
        }
    }
    out
}

pub fn py_abs(v: &PyValue) -> PyValue {
    match v {
        PyValue::Int(i) => PyValue::Int(i.abs()),
        PyValue::Float(f) => PyValue::Float(f.abs()),
        other => panic!("TypeError: bad operand for abs(): {:?}", other),
    }
}

pub fn py_sorted<T: Clone + PartialOrd>(items: &[T]) -> Vec<T> {
    let mut out = items.to_vec();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    out
}

thread_local! {
    static PY_RNG_STATE: std::cell::Cell<u64> = std::cell::Cell::new(0x853c49e6748fea9b);
}

fn py_rand_u64() -> u64 {
    PY_RNG_STATE.with(|s| {
        let mut x = s.get();
        // xorshift64*: fast, deterministic, good enough for transpiled scripts.
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s.set(x);
        x.wrapping_mul(0x2545F4914F6CDD1D)
    })
}

pub fn py_seed_int(seed: i64) {
    PY_RNG_STATE.with(|s| {
        s.set((seed as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
    });
}

pub fn py_seed() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x12345678);
    PY_RNG_STATE.with(|s| s.set(nanos | 1));
}

#[inline]
pub fn py_random() -> f64 {
    // Take the top 53 bits into an f64 mantissa: uniform in [0, 1).
    ((py_rand_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
}

#[inline]
pub fn py_randint(a: i64, b: i64) -> i64 {
    debug_assert!(a <= b, "py_randint requires a <= b");
    a + (py_rand_u64() % ((b - a + 1) as u64)) as i64
}

#[inline]
pub fn py_uniform(a: f64, b: f64) -> f64 {
    a + (b - a) * py_random()
}

#[inline]
pub fn py_choice<T: Clone>(items: &[T]) -> T {
    assert!(!items.is_empty(), "py_choice of empty sequence");
    items[(py_rand_u64() % items.len() as u64) as usize].clone()
}

pub fn py_shuffle<T>(items: &mut Vec<T>) {
    let mut i = items.len();
    while i > 1 {
        i -= 1;
        let j = (py_rand_u64() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

pub fn py_float_str(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let exp = f.abs().log10().floor() as i32;
    if (-4..16).contains(&exp) {
        if f.fract() == 0.0 {
            format!("{:.1}", f)
        } else {
            format!("{}", f)
        }
    } else {
        let mut m = f / 10f64.powi(exp);
        let mut e = exp;
        if m >= 10.0 {
            m /= 10.0;
            e += 1;
        }
        let m_str = if m.fract() == 0.0 {
            format!("{}", m as i64)
        } else {
            format!("{}", m)
        };
        let sign = if e < 0 { "-" } else { "+" };
        format!("{}e{}{:02}", m_str, sign, e.abs())
    }
}

#[inline]
pub fn py_mod_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        panic!("ZeroDivisionError: float modulo");
    }
    let r = a % b;
    if r != 0.0 && ((r > 0.0) != (b > 0.0)) {
        r + b
    } else {
        r
    }
}

pub fn py_round_bankers(x: f64) -> i64 {
    let f = x.floor();
    let d = x - f;
    if d < 0.5 {
        f as i64
    } else if d > 0.5 {
        (f + 1.0) as i64
    } else {
        let i = f as i64;
        if i % 2 == 0 { i } else { i + 1 }
    }
}

pub fn py_round_prec(x: f64, ndigits: i64) -> f64 {
    // CPython rounds the exact decimal expansion (round-half-even); Rust's
    // float formatter does the same, so go through it instead of `x * 10^n`
    // (which gets e.g. round(1.115, 2) wrong).
    if (0..=300).contains(&ndigits) && x.is_finite() {
        if let Ok(v) = format!("{:.*}", ndigits as usize, x).parse::<f64>() {
            return v;
        }
    }
    let m = 10f64.powi(ndigits as i32);
    py_round_bankers(x * m) as f64 / m
}

pub fn py_list_count<T: Clone + PartialEq>(items: &[T], val: &T) -> i64 {
    items.iter().filter(|v| *v == val).count() as i64
}

pub fn py_list_remove<T: Clone + PartialEq>(items: &mut Vec<T>, val: &T) {
    if let Some(i) = items.iter().position(|v| v == val) {
        items.remove(i);
    } else {
        panic!("ValueError: list.remove(x): x not in list");
    }
}

pub fn py_splice<T: Clone>(items: &mut Vec<T>, start: Option<i64>, stop: Option<i64>, new: Vec<T>) {
    let len = items.len() as i64;
    let norm = |bound: i64| -> usize { (if bound < 0 { len + bound } else { bound }).clamp(0, len) as usize };
    let s = norm(start.unwrap_or(0));
    let e = norm(stop.unwrap_or(len));
    let (s, e) = if s <= e { (s, e) } else { (e, e) };
    items.splice(s..e, new);
}

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
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl PyJson for i64 {
    fn py_json(&self) -> String {
        self.to_string()
    }
}

impl PyJson for f64 {
    fn py_json(&self) -> String {
        py_float_str(*self)
    }
}

impl PyJson for bool {
    fn py_json(&self) -> String {
        if *self { "true".to_string() } else { "false".to_string() }
    }
}

impl PyJson for String {
    fn py_json(&self) -> String {
        py_json_escape(self)
    }
}

impl PyJson for str {
    fn py_json(&self) -> String {
        py_json_escape(self)
    }
}

impl<T: PyJson> PyJson for Vec<T> {
    fn py_json(&self) -> String {
        let parts: Vec<String> = self.iter().map(|x| x.py_json()).collect();
        format!("[{}]", parts.join(", "))
    }
}

impl<V: PyJson + Clone> PyJson for PyDict<String, V> {
    fn py_json(&self) -> String {
        let parts: Vec<String> = self.iter().map(|(k, v)| format!("{}: {}", py_json_escape(k), v.py_json())).collect();
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

    fn err(&self, msg: &str) -> ! {
        panic!("ValueError: invalid JSON ({} at {})", msg, self.pos);
    }

    fn ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> char {
        if self.pos < self.chars.len() { self.chars[self.pos] } else { '\0' }
    }

    fn parse(&mut self) -> PyValue {
        self.ws();
        let v = self.parse_value();
        self.ws();
        v
    }

    fn parse_value(&mut self) -> PyValue {
        match self.peek() {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => PyValue::Str(self.parse_string()),
            't' => self.parse_lit("true", PyValue::Bool(true)),
            'f' => self.parse_lit("false", PyValue::Bool(false)),
            'n' => self.parse_lit("null", PyValue::None),
            c if c == '-' || c.is_ascii_digit() => self.parse_number(),
            _ => self.err("unexpected character"),
        }
    }

    fn parse_lit(&mut self, lit: &str, val: PyValue) -> PyValue {
        for c in lit.chars() {
            if self.peek() != c {
                return self.err("bad literal");
            }
            self.pos += 1;
        }
        val
    }

    fn parse_object(&mut self) -> PyValue {
        self.pos += 1; // {
        let mut map: Vec<(String, PyValue)> = Vec::new();
        self.ws();
        if self.peek() == '}' {
            self.pos += 1;
            return PyValue::Dict(map);
        }
        loop {
            self.ws();
            if self.peek() != '"' {
                return self.err("expected string key");
            }
            let k = self.parse_string();
            self.ws();
            if self.peek() != ':' {
                return self.err("expected ':'");
            }
            self.pos += 1;
            let v = self.parse();
            if let Some(slot) = map.iter_mut().find(|(ek, _)| *ek == k) {
                slot.1 = v;
            } else {
                map.push((k, v));
            }
            self.ws();
            match self.peek() {
                ',' => {
                    self.pos += 1;
                }
                '}' => {
                    self.pos += 1;
                    break;
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
        PyValue::Dict(map)
    }

    fn parse_array(&mut self) -> PyValue {
        self.pos += 1; // [
        let mut items = Vec::new();
        self.ws();
        if self.peek() == ']' {
            self.pos += 1;
            return PyValue::List(items);
        }
        loop {
            items.push(self.parse());
            self.ws();
            match self.peek() {
                ',' => {
                    self.pos += 1;
                }
                ']' => {
                    self.pos += 1;
                    break;
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
        PyValue::List(items)
    }

    fn parse_string(&mut self) -> String {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            if self.pos >= self.chars.len() {
                return self.err("unterminated string");
            }
            let c = self.chars[self.pos];
            self.pos += 1;
            match c {
                '"' => break,
                '\\' => {
                    if self.pos >= self.chars.len() {
                        return self.err("bad escape");
                    }
                    let e = self.chars[self.pos];
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
                            let mut code: u32 = 0;
                            for _ in 0..4 {
                                if self.pos >= self.chars.len() {
                                    return self.err("bad \\u escape");
                                }
                                let h = self.chars[self.pos];
                                self.pos += 1;
                                let d = h.to_digit(16).unwrap_or_else(|| self.err("bad hex"));
                                code = code * 16 + d;
                            }
                            out.push(char::from_u32(code).unwrap_or('�'));
                        }
                        _ => return self.err("bad escape"),
                    }
                }
                c => out.push(c),
            }
        }
        out
    }

    fn parse_number(&mut self) -> PyValue {
        let start = self.pos;
        if self.peek() == '-' {
            self.pos += 1;
        }
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' {
            is_float = true;
            self.pos += 1;
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.chars.len() && (self.chars[self.pos] == 'e' || self.chars[self.pos] == 'E') {
            is_float = true;
            self.pos += 1;
            if self.pos < self.chars.len() && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-') {
                self.pos += 1;
            }
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            PyValue::Float(text.parse::<f64>().unwrap_or(f64::NAN))
        } else {
            PyValue::Int(text.parse::<i64>().unwrap_or(0))
        }
    }
}

/// Dynamic attribute-style access on PyValue containers.
pub fn py_value_get(v: &PyValue, k: &PyValue, default: PyValue) -> PyValue {
    match (v, k) {
        (PyValue::Dict(map), PyValue::Str(s)) => map
            .iter()
            .find(|(ek, _)| ek == s)
            .map(|(_, vv)| vv.clone())
            .unwrap_or(default),
        (PyValue::List(items), PyValue::Int(i)) => {
            let len = items.len() as i64;
            let idx = if *i < 0 { len + i } else { *i };
            if idx >= 0 && (idx as usize) < items.len() {
                items[idx as usize].clone()
            } else {
                default
            }
        }
        _ => default,
    }
}

pub fn py_value_contains(c: &PyValue, v: &PyValue) -> bool {
    match c {
        PyValue::Dict(map) => {
            if let PyValue::Str(s) = v {
                map.iter().any(|(k, _)| k == s)
            } else {
                false
            }
        }
        PyValue::List(items) => items.iter().any(|x| x == v),
        PyValue::Str(s) => {
            if let PyValue::Str(sub) = v {
                s.contains(sub.as_str())
            } else {
                false
            }
        }
        _ => false,
    }
}

pub fn py_dyn_index(v: &PyValue, k: &PyValue) -> PyValue {
    match (v, k) {
        (PyValue::Dict(map), PyValue::Str(s)) => map
            .iter()
            .find(|(ek, _)| ek == s)
            .map(|(_, vv)| vv.clone())
            .expect("KeyError"),
        (PyValue::List(items), PyValue::Int(i)) => {
            let len = items.len() as i64;
            let idx = if *i < 0 { len + i } else { *i };
            items
                .get(idx as usize)
                .cloned()
                .expect("IndexError: list index out of range")
        }
        _ => panic!("TypeError: value is not subscriptable"),
    }
}

pub fn py_value_keys(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::Dict(map) => map.iter().map(|(k, _)| PyValue::Str(k.clone())).collect(),
        _ => Vec::new(),
    }
}

pub fn py_value_values(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::Dict(map) => map.iter().map(|(_, x)| x.clone()).collect(),
        _ => Vec::new(),
    }
}

pub fn py_value_items(v: &PyValue) -> Vec<PyValue> {
    match v {
        PyValue::Dict(map) => map
            .iter()
            .map(|(k, vv)| PyValue::List(vec![PyValue::Str(k.clone()), vv.clone()]))
            .collect(),
        _ => Vec::new(),
    }
}

pub fn py_json_loads(s: &str) -> PyValue {
    JsonParser::new(s).parse()
}

pub fn py_split_whitespace(s: &str) -> Vec<String> {
    // Python str.split() with no separator: split on whitespace runs,
    // yielding no empty strings.
    s.split_whitespace().map(|w| w.to_string()).collect()
}

fn py_stats_check(xs: &[PyValue], need: usize, name: &str) {
    if xs.len() < need {
        panic!("StatisticsError: {} requires at least {} data point(s)", name, need);
    }
}

pub fn py_mean(xs: &[PyValue]) -> f64 {
    py_stats_check(xs, 1, "mean");
    xs.iter().map(|v| v.as_f64()).sum::<f64>() / xs.len() as f64
}

pub fn py_harmonic_mean(xs: &[PyValue]) -> f64 {
    py_stats_check(xs, 1, "harmonic_mean");
    let mut denom = 0.0;
    for v in xs {
        let x = v.as_f64();
        if x == 0.0 {
            panic!("StatisticsError: harmonic_mean with zero value");
        }
        denom += 1.0 / x;
    }
    xs.len() as f64 / denom
}

fn py_sorted_idx(xs: &[PyValue]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[a].as_f64().partial_cmp(&xs[b].as_f64()).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

pub fn py_median(xs: &[PyValue]) -> PyValue {
    py_stats_check(xs, 1, "median");
    let idx = py_sorted_idx(xs);
    if xs.len() % 2 == 1 {
        // Preserve the element's own type, like Python (median([3,1,2]) == 2).
        return xs[idx[xs.len() / 2]].clone();
    }
    let (a, b) = (idx[xs.len() / 2 - 1], idx[xs.len() / 2]);
    PyValue::Float((xs[a].as_f64() + xs[b].as_f64()) / 2.0)
}

pub fn py_median_low(xs: &[PyValue]) -> PyValue {
    py_stats_check(xs, 1, "median_low");
    let idx = py_sorted_idx(xs);
    xs[idx[(xs.len() - 1) / 2]].clone()
}

pub fn py_median_high(xs: &[PyValue]) -> PyValue {
    py_stats_check(xs, 1, "median_high");
    let idx = py_sorted_idx(xs);
    xs[idx[xs.len() / 2]].clone()
}

pub fn py_mode<T: Clone + PartialEq>(xs: &[T]) -> T {
    if xs.is_empty() {
        panic!("StatisticsError: mode requires at least one data point");
    }
    let mut best = 0usize;
    let mut best_n = 0usize;
    for (i, v) in xs.iter().enumerate() {
        let n = xs.iter().filter(|w| *w == v).count();
        if n > best_n {
            best_n = n;
            best = i;
        }
    }
    xs[best].clone()
}

fn py_ss(xs: &[PyValue], c: f64) -> f64 {
    xs.iter().map(|v| { let d = v.as_f64() - c; d * d }).sum()
}

pub fn py_pvariance(xs: &[PyValue]) -> f64 {
    py_stats_check(xs, 1, "pvariance");
    py_ss(xs, py_mean(xs)) / xs.len() as f64
}

pub fn py_variance(xs: &[PyValue]) -> f64 {
    py_stats_check(xs, 2, "variance");
    py_ss(xs, py_mean(xs)) / (xs.len() - 1) as f64
}

pub fn py_pstdev(xs: &[PyValue]) -> f64 {
    py_pvariance(xs).sqrt()
}

pub fn py_stdev(xs: &[PyValue]) -> f64 {
    py_variance(xs).sqrt()
}

pub fn py_gc_collect() -> i64 {
    // Rust frees memory deterministically on scope exit; nothing to collect.
    0
}

pub fn py_gc_isenabled() -> bool {
    true
}

pub fn py_gc_disable() {}

pub fn py_gc_enable() {}

pub fn py_gc_get_count() -> (i64, i64, i64) {
    (0, 0, 0)
}

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

#[derive(Debug, Clone, Default)]
pub struct PyFile {
    path: String,
    mode: u8, // 0 = read, 1 = write (truncate), 2 = append
    wbuf: String,
    closed: bool,
}

impl PyFile {
    pub fn open_read(path: &str) -> Self {
        PyFile { path: path.to_string(), mode: 0, wbuf: String::new(), closed: false }
    }

    pub fn open_write(path: &str) -> Self {
        PyFile { path: path.to_string(), mode: 1, wbuf: String::new(), closed: false }
    }

    pub fn open_append(path: &str) -> Self {
        PyFile { path: path.to_string(), mode: 2, wbuf: String::new(), closed: false }
    }

    pub fn read(&self) -> String {
        // Match Python text mode: universal newlines.
        std::fs::read_to_string(&self.path).unwrap_or_default().replace("\r\n", "\n")
    }

    pub fn readlines(&self) -> Vec<String> {
        // Like Python: line boundaries keep their trailing '\n'.
        self.read().split_inclusive('\n').map(|s| s.to_string()).collect()
    }

    pub fn lines(&self) -> Vec<String> {
        self.readlines()
    }

    pub fn write(&mut self, s: &str) {
        if !self.closed {
            self.wbuf.push_str(s);
        }
    }

    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        if self.mode == 1 {
            let _ = std::fs::write(&self.path, &self.wbuf);
        } else if self.mode == 2 {
            let mut existing = std::fs::read_to_string(&self.path).unwrap_or_default();
            existing.push_str(&self.wbuf);
            let _ = std::fs::write(&self.path, &existing);
        }
    }
}

/// Insertion-ordered dictionary with Python semantics (linear scan;
/// no Hash/Eq bounds needed, so any key type works).
#[derive(Debug, Clone, Default)]
pub struct PyDict<K, V> {
    entries: Vec<(K, V)>,
}

impl<K: PartialEq, V> std::ops::Index<&K> for PyDict<K, V> {
    type Output = V;
    fn index(&self, k: &K) -> &V {
        self.get(k).expect("KeyError")
    }
}

impl<K: PartialEq, V> FromIterator<(K, V)> for PyDict<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = PyDict::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: PartialEq, V> PyDict<K, V> {
    pub fn new() -> Self {
        PyDict { entries: Vec::new() }
    }

    pub fn from_vec(entries: Vec<(K, V)>) -> Self {
        let mut map = PyDict::new();
        for (k, v) in entries {
            map.insert(k, v);
        }
        map
    }

    fn find(&self, k: &K) -> Option<usize> {
        self.entries.iter().position(|(ek, _)| ek == k)
    }

    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        match self.find(&k) {
            Some(i) => {
                let old = std::mem::replace(&mut self.entries[i].1, v);
                Some(old)
            }
            None => {
                self.entries.push((k, v));
                None
            }
        }
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.find(k).map(|i| &self.entries[i].1)
    }

    pub fn remove(&mut self, k: &K) -> Option<V> {
        self.find(k).map(|i| self.entries.remove(i).1)
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.find(k).is_some()
    }

    pub fn entry(&mut self, k: K) -> PyDictEntry<'_, K, V> {
        PyDictEntry { map: self, key: k }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (K, V)> {
        self.entries.iter()
    }

    pub fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }

    pub fn into_vec(self) -> Vec<(K, V)> {
        self.entries
    }
}

impl<K: PartialEq + Clone, V: Clone> PyDict<K, V> {
    pub fn keys(&self) -> Vec<K> {
        self.entries.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn values(&self) -> Vec<V> {
        self.entries.iter().map(|(_, v)| v.clone()).collect()
    }

    pub fn items(&self) -> Vec<(K, V)> {
        self.entries.clone()
    }
}

impl<K: PartialEq, V: PartialEq> PartialEq for PyDict<K, V> {
    fn eq(&self, other: &Self) -> bool {
        // Order-insensitive, like Python dict ==.
        self.len() == other.len()
            && self.entries.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

pub struct PyDictEntry<'a, K, V> {
    map: &'a mut PyDict<K, V>,
    key: K,
}

impl<'a, K: PartialEq, V> PyDictEntry<'a, K, V> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        let i = match self.map.find(&self.key) {
            Some(i) => i,
            None => {
                self.map.entries.push((self.key, default));
                self.map.entries.len() - 1
            }
        };
        &mut self.map.entries[i].1
    }
}

pub fn py_unix_time() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

static PY_T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

pub fn py_perf_counter() -> f64 {
    PY_T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64()
}

pub fn py_input(prompt: &str) -> String {
    if !prompt.is_empty() {
        print!("{}", prompt);
        let _ = io::stdout().flush();
    }
    let mut buf = String::new();
    let _ = io::stdin().read_line(&mut buf);
    buf.trim_end_matches(&['\r', '\n'][..]).to_string()
}
"""
