"""Regenerates the README graphics (light + dark variants of each).

    python assets/generate.py

Benchmark numbers live in BENCH below; update them from
`pyrust run benchmarks/numpy_vs_hybrid.py` and `python benchmarks/numpy_vs_hybrid.py`.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

THEMES = {
    "light": {
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8983", "line": "#d9d8d2",
        "card": "#f6f5f2", "py": "#2a78d6", "rust": "#eb6834", "neutral": "#b9b8b2",
        "chip": "#ffffff",
    },
    "dark": {
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#8a8983", "line": "#34332f",
        "card": "#1a1a19", "py": "#3987e5", "rust": "#d95926", "neutral": "#5f5e59",
        "chip": "#242422",
    },
}

# (workload, subtitle, [(label, milliseconds, highlighted)])
BENCH = [
    ("Mandelbrot", "400 x 400 grid, 200 iterations", [
        ("Plain Python", 1325, False), ("NumPy", 1134, False), ("Numba", 17, False), ("PyRust", 16, True)]),
    ("Collatz", "longest chain below 300,000", [
        ("Plain Python", 3193, False), ("NumPy", 1594, False), ("Numba", 28, False), ("PyRust", 76, True)]),
    ("Sum of i*i % 7", "5,000,000 integers", [
        ("Plain Python", 452, False), ("NumPy", 86, False), ("Numba", 4, False), ("PyRust", 8, True)]),
]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w: int, h: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{esc(title)}">\n<title>{esc(title)}</title>\n{body}</svg>\n'
    )


def banner(t: dict) -> str:
    w, h = 880, 250
    # The whole pitch in one keystroke: `script.py` + `r`.
    x0, y0, size = 440, 118, 78
    char = size * 0.6
    word = "script.py"
    total = (len(word) + 1) * char
    left = x0 - total / 2
    rx = left + len(word) * char
    body = f"""<style>
  .cur {{ animation: blink 1.15s steps(1) infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0.25; }} }}
  @media (prefers-reduced-motion: reduce) {{ .cur {{ animation: none; }} }}
</style>
<text x="{rx - 4:.1f}" y="{y0}" text-anchor="end" font-family="{MONO}" font-size="{size}" font-weight="700" fill="{t['ink']}" xml:space="preserve"><tspan fill="{t['muted']}">script</tspan><tspan fill="{t['muted']}">.</tspan><tspan fill="{t['py']}">py</tspan></text>
<rect class="cur" x="{rx + 2:.1f}" y="{y0 - size * 0.80:.1f}" width="{char - 4:.1f}" height="{size * 1.02:.1f}" rx="6" fill="{t['rust']}"/>
<text x="{rx + char / 2:.1f}" y="{y0}" text-anchor="middle" font-family="{MONO}" font-size="{size}" font-weight="700" fill="#ffffff">r</text>
<text x="{x0}" y="{y0 + 56}" text-anchor="middle" font-family="{SANS}" font-size="24" font-weight="600" fill="{t['ink']}">Add one letter. Keep your Python.</text>
<text x="{x0}" y="{y0 + 88}" text-anchor="middle" font-family="{SANS}" font-size="17" fill="{t['ink2']}">Native speed where your code allows it. Plain Python everywhere else.</text>
"""
    return svg(w, h, body, "PyRust: rename script.py to script.pyr. Add one letter, keep your Python.")


def bar_path(x: float, y: float, width: float, height: float) -> str:
    r = min(4.0, width / 2, height / 2)
    return (
        f"M{x:.1f},{y:.1f} h{width - r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} "
        f"v{height - 2 * r:.1f} a{r:.1f},{r:.1f} 0 0 1 -{r:.1f},{r:.1f} h-{width - r:.1f} z"
    )


def bench(t: dict) -> str:
    w = 880
    label_x, bar_x, bar_w = 150, 162, 500
    row_h, bar_h, panel_gap = 26, 14, 34
    parts: list[str] = []
    y = 8
    for name, sub, rows in BENCH:
        y += 20
        parts.append(
            f'<text x="16" y="{y}" font-family="{SANS}" font-size="15" font-weight="600" fill="{t["ink"]}">{esc(name)}'
            f'<tspan font-weight="400" fill="{t["muted"]}" dx="10" font-size="13">{esc(sub)}</tspan></text>'
        )
        y += 12
        slowest = max(ms for _, ms, _ in rows)
        base = rows[0][1]
        parts.append(f'<line x1="{bar_x}" y1="{y}" x2="{bar_x}" y2="{y + row_h * len(rows)}" stroke="{t["line"]}" stroke-width="1"/>')
        for label, ms, hot in rows:
            mid = y + row_h / 2
            width = max(3.0, bar_w * ms / slowest)
            fill = t["rust"] if hot else t["neutral"]
            weight = "600" if hot else "400"
            parts.append(
                f'<text x="{label_x}" y="{mid + 4.5:.1f}" text-anchor="end" font-family="{SANS}" font-size="13" '
                f'font-weight="{weight}" fill="{t["ink"] if hot else t["ink2"]}">{esc(label)}</text>'
            )
            parts.append(f'<path d="{bar_path(bar_x, mid - bar_h / 2, width, bar_h)}" fill="{fill}"/>')
            note = f"{ms:,} ms"
            if label != "Plain Python":
                ratio = base / ms
                note += f"   {ratio:.0f}x faster" if ratio >= 10 else f"   {ratio:.1f}x faster"
            parts.append(
                f'<text x="{bar_x + width + 8:.1f}" y="{mid + 4.5:.1f}" font-family="{MONO}" font-size="12.5" '
                f'font-weight="{weight}" fill="{t["ink"] if hot else t["ink2"]}" xml:space="preserve">{esc(note)}</text>'
            )
            y += row_h
        y += panel_gap - 20
    h = y + 6
    return svg(w, h, "\n".join(parts) + "\n", "Benchmark: time in milliseconds for plain Python, NumPy, Numba and PyRust on three workloads. Shorter is faster.")


def tiers(t: dict) -> str:
    w, h = 880, 292
    cards = [
        ("NATIVE", t["rust"], "the program fits the native subset",
         ["Entire program becomes", "one Rust binary."], "tens of times faster", 1.0),
        ("HYBRID", t["py"], "it uses any library or any syntax",
         ["Hot functions become native.", "The rest runs on CPython."], "hot paths 40-80x faster", 0.62),
        ("PYTHON", t["neutral"], "there is nothing to compile",
         ["Runs on CPython,", "exactly as before."], "unchanged, ~30 ms launch cost", 0.12),
    ]
    cw, gap, top = 272, 22, 66
    left = (w - (3 * cw + 2 * gap)) / 2
    parts = [
        f'<rect x="{w / 2 - 92}" y="6" width="184" height="34" rx="17" fill="{t["chip"]}" stroke="{t["line"]}"/>',
        f'<text x="{w / 2}" y="28" text-anchor="middle" font-family="{MONO}" font-size="14" font-weight="600" fill="{t["ink"]}">pyrust run app.pyr</text>',
    ]
    for i, (name, color, when, what, speed, frac) in enumerate(cards):
        x = left + i * (cw + gap)
        cx = x + cw / 2
        parts.append(f'<path d="M{w / 2:.1f},40 C{w / 2:.1f},56 {cx:.1f},48 {cx:.1f},{top}" fill="none" stroke="{t["line"]}" stroke-width="1.5"/>')
        parts.append(f'<rect x="{x:.1f}" y="{top}" width="{cw}" height="214" rx="12" fill="{t["card"]}" stroke="{t["line"]}"/>')
        parts.append(f'<rect x="{x + 20:.1f}" y="{top + 20}" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 38:.1f}" y="{top + 30}" font-family="{MONO}" font-size="14" font-weight="700" letter-spacing="1.5" fill="{t["ink"]}">{name}</text>')
        parts.append(f'<text x="{x + 20:.1f}" y="{top + 56}" font-family="{SANS}" font-size="12.5" fill="{t["muted"]}">when {esc(when)}</text>')
        for j, line in enumerate(what):
            parts.append(f'<text x="{x + 20:.1f}" y="{top + 92 + j * 22}" font-family="{SANS}" font-size="15.5" font-weight="600" fill="{t["ink"]}">{esc(line)}</text>')
        parts.append(f'<rect x="{x + 20:.1f}" y="{top + 152}" width="{cw - 40}" height="6" rx="3" fill="{t["line"]}"/>')
        parts.append(f'<rect x="{x + 20:.1f}" y="{top + 152}" width="{(cw - 40) * frac:.1f}" height="6" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + 20:.1f}" y="{top + 182}" font-family="{SANS}" font-size="13" fill="{t["ink2"]}">{esc(speed)}</text>')
    return svg(w, h, "\n".join(parts) + "\n", "pyrust run picks one of three tiers: native, hybrid, or plain Python. Output is identical in all three.")


if __name__ == "__main__":
    for mode, theme in THEMES.items():
        for name, fn in (("banner", banner), ("bench", bench), ("tiers", tiers)):
            out = HERE / f"{name}-{mode}.svg"
            out.write_text(fn(theme), encoding="utf-8")
            print("wrote", out.name)
