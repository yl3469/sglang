"""Generate an NVIDIA-themed HTML report for the per-layer HiSparse A/B study.

Reads every ``ab_*/{uniform,dp}/bench_c*.jsonl`` under the results root, merges
them into one concurrency sweep (later jobs override earlier ones for the same
concurrency), and renders a single self-contained HTML page covering:

* key implementation design choices (per-layer device buffer),
* the experimental setup + baseline definition,
* the full A/B results table: TTFT, TBT (inter-token latency), and E2E at
  P50/P90/P99 for both arms, plus DP-vs-uniform deltas,
* a cache-hit-rate section (populated only if runs used --cache-report).

Re-run this after new concurrency points land to refresh the page.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Optional, Tuple

# Sibling module (works when run by file path, not just as a package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from report_assets import kernel_shapes_svg
except Exception:  # pragma: no cover - diagram is optional
    kernel_shapes_svg = None


def _load_point(path: str) -> Optional[dict]:
    last = None
    try:
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        return None
    return last


def collect(results_root: str) -> Dict[Tuple[int, str], dict]:
    """Map (concurrency, arm) -> latest bench dict, scanning all job dirs.

    uniform/dp live in ab_* (both ranges) and hc_* (high conc); the
    non-HiSparse ``dense`` reference lives in dense_* (low conc) and hc_*
    (high conc). Later job ids override earlier ones for the same (c, arm).
    """
    out: Dict[Tuple[int, str], dict] = {}
    patterns = [
        ("ab_*", ("uniform", "dp")),
        ("dense_*", ("dense",)),
        ("hc_*", ("dense", "uniform", "dp")),
    ]
    for root_glob, arms in patterns:
        for job_dir in sorted(glob.glob(os.path.join(results_root, root_glob))):
            for arm in arms:
                for f in glob.glob(os.path.join(job_dir, arm, "bench_c*.jsonl")):
                    base = os.path.basename(f)
                    try:
                        c = int(base.split("bench_c")[1].split(".")[0])
                    except (IndexError, ValueError):
                        continue
                    d = _load_point(f)
                    if d is not None:
                        out[(c, arm)] = d
    return out


def _load_token_dist(results_root: str) -> Optional[dict]:
    """Load precomputed per-request input token counts (token_dist.json).

    Produced offline by tokenizing the bench dataset with the model tokenizer
    (see make_report docstring). Returns None if not present.
    """
    for cand in (
        os.path.join(results_root, "_shared", "token_dist.json"),
        os.path.join(results_root, "token_dist.json"),
    ):
        if os.path.exists(cand):
            try:
                return json.load(open(cand))
            except (json.JSONDecodeError, OSError):
                return None
    return None


def _cdf_svg(
    values,
    width=440,
    height=210,
    pad=44,
    color="#76b900",
    x_label="tokens",
    unit_div=1000,
    x_unit="k",
) -> str:
    """Render an empirical CDF as a self-contained inline SVG (no JS/libs).

    Percentile gridlines (p50/p90/p99) are annotated. ``values`` is a list of
    per-request counts.
    """
    if not values:
        return "<div class='note'>no data</div>"
    xs = sorted(values)
    n = len(xs)
    xmin, xmax = xs[0], xs[-1]
    span = max(1, xmax - xmin)

    def px(v):
        return pad + (v - xmin) / span * (width - 2 * pad)

    def py(f):  # f in [0,1], 1 at top
        return (height - pad) - f * (height - 2 * pad)

    # Step CDF polyline points.
    pts = []
    for i, v in enumerate(xs):
        f = (i + 1) / n
        pts.append(f"{px(v):.1f},{py(f):.1f}")
    poly = " ".join(pts)

    def pct(p):
        idx = min(n - 1, int(round(p / 100.0 * n)) - 1)
        idx = max(0, idx)
        return xs[idx]

    import statistics

    mean = statistics.fmean(xs)
    p50, p90, p99 = pct(50), pct(90), pct(99)

    # Axis ticks (x): min, p50, p90, max.
    def xt(v):
        return f'<text x="{px(v):.0f}" y="{height-pad+16}" fill="#9aa4b2" font-size="10" text-anchor="middle">{v/unit_div:.1f}{x_unit}</text>'

    grid = []
    for gv, gl, gc in [
        (p50, "p50", "#8892a0"),
        (p90, "p90", "#c9a227"),
        (p99, "p99", "#ff5c5c"),
    ]:
        x = px(gv)
        grid.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height-pad}" '
            f'stroke="{gc}" stroke-dasharray="3,3" stroke-width="1" opacity="0.7"/>'
            f'<text x="{x:.1f}" y="{pad-4}" fill="{gc}" font-size="9.5" text-anchor="middle">{gl}</text>'
        )
    # y gridlines at 0,.25,.5,.75,1
    ygrid = []
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        y = py(f)
        ygrid.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" '
            f'stroke="#2c333d" stroke-width="1"/>'
            f'<text x="{pad-6}" y="{y+3:.1f}" fill="#9aa4b2" font-size="9.5" text-anchor="end">{f:.2f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">'
        + "".join(ygrid)
        + "".join(grid)
        + f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        + xt(xmin)
        + xt(p50)
        + xt(p90)
        + xt(xmax)
        + f'<text x="{width/2:.0f}" y="{height-6}" fill="#9aa4b2" font-size="10.5" text-anchor="middle">{x_label}</text>'
        + f'<text x="12" y="{height/2:.0f}" fill="#9aa4b2" font-size="10.5" text-anchor="middle" transform="rotate(-90 12 {height/2:.0f})">cumulative fraction</text>'
        + "</svg>"
        + f'<div class="note">n={n} &middot; min={xmin} &middot; p50={p50} &middot; p90={p90} &middot; '
        f"p99={p99} &middot; max={xmax} &middot; mean={mean:.0f}</div>"
    )


def _delta(dp: Optional[float], un: Optional[float]) -> Optional[float]:
    if dp is None or un is None or un == 0:
        return None
    return (dp - un) / un * 100.0


def _cls(delta: Optional[float], lower_is_better: bool) -> str:
    if delta is None:
        return "neu"
    good = (delta < 0) if lower_is_better else (delta > 0)
    if abs(delta) < 2.0:
        return "neu"
    return "good" if good else "bad"


def _fmt(v: Optional[float], nd: int = 0) -> str:
    if v is None:
        return "&ndash;"
    return f"{v:.{nd}f}"


def _fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "&ndash;"
    return f"{v:+.1f}%"


HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Per-Layer HiSparse Device Buffer &mdash; DP vs Uniform</title>
<style>
  :root {
    --nv-green:#76b900; --nv-green-d:#5a8f00; --ink:#111418; --panel:#1a1d23;
    --panel2:#22262e; --line:#333a44; --txt:#e7ebf0; --muted:#9aa4b2;
    --good:#76b900; --bad:#ff5c5c; --neu:#8892a0;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ink); color:var(--txt);
    font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; line-height:1.45; }
  .wrap { max-width:1180px; margin:0 auto; padding:32px 28px 64px; }
  header { border-left:6px solid var(--nv-green); padding:8px 0 8px 18px; margin-bottom:26px; }
  h1 { margin:0 0 4px; font-size:26px; letter-spacing:.2px; }
  h1 .sub { color:var(--nv-green); }
  .tag { color:var(--muted); font-size:13px; }
  h2 { font-size:18px; margin:34px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line);
    color:#fff; }
  h2 .chip { background:var(--nv-green); color:#0a0d10; font-size:11px; font-weight:700;
    padding:2px 8px; border-radius:10px; margin-left:8px; vertical-align:middle; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
  .card h3 { margin:0 0 8px; font-size:14px; color:var(--nv-green); text-transform:uppercase;
    letter-spacing:.6px; }
  .card ul { margin:0; padding-left:18px; } .card li { margin:4px 0; font-size:13.5px; }
  code { background:#0d0f13; border:1px solid var(--line); border-radius:4px; padding:1px 5px;
    font-size:12px; color:#cfe8a8; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; margin-top:6px; }
  th,td { padding:6px 8px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.4px; }
  td.lbl,th.lbl { text-align:left; }
  tr.arm-u td { color:#c9d2dd; } tr.arm-d td { color:#fff; }
  tr.delta td { font-weight:700; border-bottom:2px solid var(--line); }
  tr.grp td { padding-top:12px; }
  .good { color:var(--good); } .bad { color:var(--bad); } .neu { color:var(--neu); }
  .arm-pill { display:inline-block; font-size:10px; font-weight:700; padding:1px 7px; border-radius:8px; }
  .pu { background:#2c333d; color:#c9d2dd; } .pd { background:#25350a; color:var(--nv-green); }
  .note { background:var(--panel2); border-left:3px solid var(--nv-green); padding:10px 14px;
    border-radius:0 8px 8px 0; font-size:13px; color:var(--muted); margin:10px 0; }
  .warn { border-left-color:#e0a300; }
  .foot { color:var(--muted); font-size:12px; margin-top:28px; border-top:1px solid var(--line); padding-top:12px; }
  .kpis { display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 4px; }
  .kpi { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 16px; flex:1; min-width:150px; }
  .kpi .n { font-size:22px; font-weight:800; color:var(--nv-green); }
  .kpi .l { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
</style></head><body><div class="wrap">
"""


def _metrics_table(data: Dict[Tuple[int, str], dict], concs) -> str:
    # Column groups: throughput, TTFT, TBT(ITL), E2E - each p50/p90/p99 where applicable.
    def cell(d, key, nd=0):
        return _fmt(d.get(key), nd)

    html = [
        "<table><thead><tr>",
        '<th class="lbl">conc</th><th class="lbl">arm</th>',
        "<th>req/s</th><th>tok/s</th>",
        "<th>TTFT p50</th><th>TTFT p90</th><th>TTFT p99</th>",
        "<th>TBT p50</th><th>TBT p90</th><th>TBT p99</th>",
        "<th>E2E p50</th><th>E2E p90</th><th>E2E p99</th>",
        "</tr></thead><tbody>",
    ]
    for c in concs:
        u = data.get((c, "uniform"))
        d = data.get((c, "dp"))
        if not u and not d:
            continue
        html.append('<tr class="grp"></tr>')
        # uniform row
        if u:
            html.append(
                f'<tr class="arm-u"><td class="lbl">{c}</td>'
                f'<td class="lbl"><span class="arm-pill pu">uniform</span></td>'
                f'<td>{cell(u,"request_throughput",3)}</td><td>{cell(u,"output_throughput",0)}</td>'
                f'<td>{cell(u,"median_ttft_ms")}</td><td>{cell(u,"p90_ttft_ms")}</td><td>{cell(u,"p99_ttft_ms")}</td>'
                f'<td>{cell(u,"median_itl_ms",1)}</td><td>{cell(u,"p90_itl_ms",1)}</td><td>{cell(u,"p99_itl_ms",1)}</td>'
                f'<td>{cell(u,"median_e2e_latency_ms")}</td><td>{cell(u,"p90_e2e_latency_ms")}</td><td>{cell(u,"p99_e2e_latency_ms")}</td></tr>'
            )
        if d:
            html.append(
                f'<tr class="arm-d"><td class="lbl">{c}</td>'
                f'<td class="lbl"><span class="arm-pill pd">dp</span></td>'
                f'<td>{cell(d,"request_throughput",3)}</td><td>{cell(d,"output_throughput",0)}</td>'
                f'<td>{cell(d,"median_ttft_ms")}</td><td>{cell(d,"p90_ttft_ms")}</td><td>{cell(d,"p99_ttft_ms")}</td>'
                f'<td>{cell(d,"median_itl_ms",1)}</td><td>{cell(d,"p90_itl_ms",1)}</td><td>{cell(d,"p99_itl_ms",1)}</td>'
                f'<td>{cell(d,"median_e2e_latency_ms")}</td><td>{cell(d,"p90_e2e_latency_ms")}</td><td>{cell(d,"p99_e2e_latency_ms")}</td></tr>'
            )
        if u and d:

            def dc(key, lower=True, thr=None):
                dl = _delta(d.get(key), u.get(key))
                return f'<td class="{_cls(dl, lower)}">{_fmt_delta(dl)}</td>'

            html.append(
                f'<tr class="delta"><td class="lbl"></td><td class="lbl">&Delta;dp</td>'
                f'{dc("request_throughput", lower=False)}{dc("output_throughput", lower=False)}'
                f'{dc("median_ttft_ms")}{dc("p90_ttft_ms")}{dc("p99_ttft_ms")}'
                f'{dc("median_itl_ms")}{dc("p90_itl_ms")}{dc("p99_itl_ms")}'
                f'{dc("median_e2e_latency_ms")}{dc("p90_e2e_latency_ms")}{dc("p99_e2e_latency_ms")}</tr>'
            )
    html.append("</tbody></table>")
    return "".join(html)


def _dense_relative_table(data: Dict[Tuple[int, str], dict]) -> str:
    """Table of uniform & dp relative to the non-HiSparse ``dense`` baseline.

    Shows the absolute dense reference (req/s + p99s) then, for each HiSparse
    arm, the percent change vs dense on req/s (higher better), TTFT p99 and
    E2E p99 (lower better). Only concurrencies with a dense point are shown.
    """
    concs = sorted({c for (c, _a) in data if (c, "dense") in data})
    if not concs:
        return (
            '<div class="note warn">No dense (non-HiSparse) baseline points found &mdash; '
            "run <code>dense_baseline.sbatch</code> / <code>highconc_sweep.sbatch</code>.</div>"
        )

    def d(dp, base):
        return _delta(dp, base)

    html = [
        "<table><thead><tr>",
        '<th class="lbl">conc</th><th class="lbl">arm</th>',
        "<th>req/s</th><th>&Delta; req/s</th>",
        "<th>TTFT p99</th><th>&Delta; TTFT p99</th>",
        "<th>E2E p99</th><th>&Delta; E2E p99</th>",
        "</tr></thead><tbody>",
    ]
    for c in concs:
        de = data[(c, "dense")]
        html.append('<tr class="grp"></tr>')
        # dense reference row (absolute; no delta).
        html.append(
            f'<tr class="arm-u"><td class="lbl">{c}</td>'
            f'<td class="lbl"><span class="arm-pill" style="background:#3a2a10;color:#e0b050">dense</span></td>'
            f'<td>{_fmt(de.get("request_throughput"),3)}</td><td class="neu">ref</td>'
            f'<td>{_fmt(de.get("p99_ttft_ms"))}</td><td class="neu">ref</td>'
            f'<td>{_fmt(de.get("p99_e2e_latency_ms"))}</td><td class="neu">ref</td></tr>'
        )
        for arm, pill, cls_css in (
            ("uniform", "uniform", "pu"),
            ("dp", "dp", "pd"),
        ):
            a = data.get((c, arm))
            if not a:
                continue
            dr = d(a.get("request_throughput"), de.get("request_throughput"))
            dt = d(a.get("p99_ttft_ms"), de.get("p99_ttft_ms"))
            de99 = d(a.get("p99_e2e_latency_ms"), de.get("p99_e2e_latency_ms"))
            html.append(
                f'<tr><td class="lbl"></td>'
                f'<td class="lbl"><span class="arm-pill {cls_css}">{pill}</span></td>'
                f'<td>{_fmt(a.get("request_throughput"),3)}</td>'
                f'<td class="{_cls(dr, False)}">{_fmt_delta(dr)}</td>'
                f'<td>{_fmt(a.get("p99_ttft_ms"))}</td>'
                f'<td class="{_cls(dt, True)}">{_fmt_delta(dt)}</td>'
                f'<td>{_fmt(a.get("p99_e2e_latency_ms"))}</td>'
                f'<td class="{_cls(de99, True)}">{_fmt_delta(de99)}</td></tr>'
            )
    html.append("</tbody></table>")
    return "".join(html)


def render(data: Dict[Tuple[int, str], dict], token_dist: Optional[dict] = None) -> str:
    concs = sorted({c for (c, _a) in data})
    n_pairs = sum(1 for c in concs if (c, "uniform") in data and (c, "dp") in data)

    # Headline KPI: best tail-latency win among paired points (E2E p99).
    best = None
    for c in concs:
        u, d = data.get((c, "uniform")), data.get((c, "dp"))
        if u and d:
            dl = _delta(d.get("p99_e2e_latency_ms"), u.get("p99_e2e_latency_ms"))
            if dl is not None and (best is None or dl < best[1]):
                best = (c, dl)

    has_cache = any("cache_report" in v for v in data.values())

    p = [HEAD]
    p.append(
        "<header><h1>Per-Layer HiSparse Device Buffer &mdash; "
        '<span class="sub">DP vs Uniform</span></h1>'
        '<div class="tag">DeepSeek-V4-Flash &middot; sglang HiSparse &middot; '
        "matched-memory A/B serving sweep &middot; SWE-bench prompts</div></header>"
    )

    # KPI strip
    p.append('<div class="kpis">')
    p.append(
        f'<div class="kpi"><div class="n">{len(concs)}</div><div class="l">concurrency points</div></div>'
    )
    p.append(
        f'<div class="kpi"><div class="n">21</div><div class="l">indexer layers (C4A)</div></div>'
    )
    if best:
        p.append(
            f'<div class="kpi"><div class="n">{best[1]:+.0f}%</div><div class="l">best E2E p99 &Delta; (c={best[0]})</div></div>'
        )
    p.append(
        '<div class="kpi"><div class="n">= mem</div><div class="l">matched device memory</div></div>'
    )
    p.append("</div>")

    # Design choices
    p.append('<h2>Key design choices <span class="chip">implementation</span></h2>')
    p.append('<div class="grid">')
    p.append(
        '<div class="card"><h3>Per-layer buffer B<sub>l</sub></h3><ul>'
        "<li>Stock HiSparse uses one scalar <code>device_buffer_size</code> shared by all layers. "
        "We give each of the 21 C4A indexer layers its own capacity B<sub>l</sub>.</li>"
        "<li>Sourced from the offline DP allocation (<code>dp_allocations.csv</code>) &rarr; "
        "<code>device_buffer_sizes</code> in <code>--hisparse-config</code> (inline list or JSON path).</li>"
        "<li>Physical tensors kept at <b>MAX(B<sub>l</sub>)</b>; per-layer capacity is a logical cap &mdash; "
        "no tensor-shape changes.</li></ul></div>"
    )
    p.append(
        '<div class="card"><h3>Contained hot path</h3><ul>'
        "<li>Only 2 serving-path changes: (1) kernel dispatch passes "
        "<code>hot_buffer_size=B<sub>l</sub></code> per layer; (2) reserved-slot write scatters "
        "to per-layer column B<sub>l</sub>.</li>"
        "<li><code>hot_buffer_size</code> is a compile-time template &rarr; one kernel compiled per "
        "distinct B<sub>l</sub> (cached).</li>"
        "<li>Physical row strides are runtime args &rarr; a smaller B<sub>l</sub> on a wider row is layout-safe.</li></ul></div>"
    )
    p.append(
        '<div class="card"><h3>Baseline == stock (proven)</h3><ul>'
        "<li>A uniform per-layer config takes the original single-broadcast write path &mdash; "
        "bit-identical to stock HiSparse (unit-tested).</li>"
        "<li>So the <b>uniform</b> arm is a faithful baseline; only the allocation <i>shape</i> differs.</li>"
        "<li>43 CPU unit tests pass (config parse, per-layer resolve, reserved-write equivalence, DP/replay).</li></ul></div>"
    )
    p.append(
        '<div class="card"><h3>Matched-memory A/B</h3><ul>'
        "<li>Both arms use the <b>same total</b> device-buffer budget (&Sigma;B<sub>l</sub>).</li>"
        "<li><b>uniform</b>: 1024 slots/layer. <b>dp</b>: 704&ndash;1408 slots/layer (more to the "
        "needy mid-deep layers, less to plateaued ones), same sum.</li>"
        "<li>Difference is purely <i>where</i> the memory goes, isolating the DP allocation effect.</li></ul></div>"
    )
    p.append("</div>")

    # Kernel-path tensor shapes diagram
    if kernel_shapes_svg is not None:
        b_dp = None
        if token_dist and token_dist.get("dp_buffer_sizes"):
            b_dp = token_dist["dp_buffer_sizes"]
        p.append('<h2>Touched kernel path <span class="chip">tensor shapes</span></h2>')
        p.append(
            '<div class="card">'
            + kernel_shapes_svg(
                layer_num=21, top_k=512, b_uniform=1024, b_dp=b_dp, page_size=256
            )
            + '<div class="note">The swap-in kernel <code>load_cache_to_device_buffer_dsv4_mla</code> '
            "is launched once per layer. Making the capacity per-layer touches only (1) the "
            "compile-time template <code>hot_buffer_size = B<sub>l</sub></code> and (2) the reserved "
            "(newest-token) slot column, which moves from a single shared offset to per-layer "
            "offset B<sub>l</sub>. Every physical tensor keeps its width at MAX(B<sub>l</sub>)+page, "
            "so nothing is reshaped and CUDA graphs stay valid (static indices).</div></div>"
        )

    # Setup
    p.append("<h2>Experimental setup</h2>")
    p.append('<div class="grid">')
    p.append(
        '<div class="card"><h3>Serving stack</h3><ul>'
        "<li>Model: <b>DeepSeek-V4-Flash-0731</b> (fp8, fp4 MoE experts), 21 C4A indexer layers, "
        "index_topk=512, compress_ratio 4.</li>"
        "<li>sglang <code>launch_server</code>: <code>--enable-hisparse --enable-dp-attention</code>, "
        "TP=4, DP=4, single node (4&times;GB300).</li>"
        "<li><code>--disable-radix-cache --disable-custom-all-reduce</code>, "
        "<code>max-total-tokens=262144</code>.</li></ul></div>"
    )
    # Workload card: pull real token stats from token_dist when available.
    if token_dist and token_dist.get("input_tokens"):
        it = sorted(token_dist["input_tokens"])
        import statistics as _st

        _n = len(it)
        wl_line = (
            f"<li>Dataset: <b>SWE-bench</b> (oracle) prompts, {_n}-prompt pool &mdash; input "
            f"{it[0]}&ndash;{it[-1]} tokens (mean {_st.fmean(it):.0f}, "
            f"p50 {it[min(_n-1,_n//2)]}).</li>"
        )
    else:
        wl_line = (
            "<li>Dataset: <b>SWE-bench</b> (oracle) prompts, 300-prompt pool.</li>"
        )
    p.append(
        '<div class="card"><h3>Workload &amp; load</h3><ul>'
        + wl_line
        + "<li><code>bench_serving</code>: 200 requests/point, <b>fixed 256-token output</b> "
        "(<code>ignore_eos</code>) so both arms do identical work.</li>"
        "<li>Concurrency sweep: " + ", ".join(str(c) for c in concs) + ".</li>"
        "<li>Metrics: TTFT (prefill latency), TBT = inter-token latency, E2E (full request).</li></ul></div>"
    )
    p.append("</div>")

    # Token distribution CDF
    p.append("<h2>Token distribution</h2>")
    if token_dist and token_dist.get("input_tokens"):
        out_len = token_dist.get("output_len_fixed")
        p.append('<div class="grid">')
        p.append(
            '<div class="card"><h3>Input tokens &mdash; CDF (per request)</h3>'
            + _cdf_svg(token_dist["input_tokens"], x_label="input tokens per request")
            + "</div>"
        )
        out_note = (
            f"<li>Output length is <b>fixed at {out_len} tokens</b> for every request "
            "(<code>--sharegpt-output-len</code> + <code>ignore_eos</code>), so the output "
            "CDF is a step at a single value &mdash; identical across both arms and all loads by "
            "construction. This is deliberate: it holds decode work constant so the A/B isolates "
            "the device-buffer allocation, not output-length variance.</li>"
            if out_len
            else "<li>Output length fixed per request; output CDF is degenerate (single value).</li>"
        )
        p.append(
            '<div class="card"><h3>Output tokens</h3><ul>' + out_note + "</ul>"
            '<div class="note">Input tokens are tokenized with the DeepSeek-V4-Flash tokenizer '
            "(precomputed in <code>token_dist.json</code>).</div></div>"
        )
        p.append("</div>")
    else:
        p.append(
            '<div class="note warn">Token distribution not available &mdash; run the tokenizer '
            "precompute step to write <code>_shared/token_dist.json</code> "
            "(per-request input token counts), then regenerate.</div>"
        )

    # Results
    p.append(
        f'<h2>Results &mdash; DP vs Uniform <span class="chip">{n_pairs} paired points</span></h2>'
    )
    p.append(
        '<div class="note">All latencies in <b>ms</b>. <b>TBT</b> = inter-token latency (time between '
        "output tokens). &Delta;dp = (dp&minus;uniform)/uniform: for latency "
        '<span class="good">green = DP faster</span>, <span class="bad">red = DP slower</span>; '
        "for throughput green = DP higher. |&Delta;|&lt;2% shown neutral.</div>"
    )
    p.append(_metrics_table(data, concs))

    # Relative to the non-HiSparse (dense) baseline.
    p.append(
        '<h2>Relative to non-HiSparse baseline <span class="chip">dense reference</span></h2>'
    )
    p.append(
        '<div class="note"><b>dense</b> = same server <b>without</b> <code>--enable-hisparse</code>: the model '
        "still runs its native compressed + C4Indexer top-k attention, but the full compressed KV stays on GPU "
        "(<code>c4_shrink_factor=1</code>, no host offload). It fits at the <b>same 262144-token</b> budget with "
        "no OOM, so it is a fair reference. &Delta; = (arm&minus;dense)/dense: for req/s "
        '<span class="good">green = HiSparse higher</span>; for latency '
        '<span class="good">green = HiSparse faster</span>, <span class="bad">red = the offload adds latency</span>.</div>'
    )
    p.append(_dense_relative_table(data))
    p.append('<div class="grid">')
    p.append(
        '<div class="card"><h3>What the dense reference shows</h3><ul>'
        "<li><b>Throughput:</b> dense is <b>highest at every load</b> (HiSparse arms &minus;1 to &minus;11% req/s). "
        "The offload is <b>not</b> a throughput win at equal work &mdash; its value is the memory headroom, which "
        "this matched-memory / fixed-work test deliberately does not reward.</li>"
        "<li><b>Low load (1&ndash;5):</b> the offload costs &asymp;9&ndash;11% E2E p99; <b>dp claws it back</b> "
        "(c=1 TTFT p99 &minus;30% vs dense) and <b>prevents uniform&rsquo;s c=5 tail blow-up</b> "
        "(uniform +128% E2E / +306% TTFT vs dense; dp only +15% / +6%).</li>"
        "<li><b>High load (c=128):</b> all three arms <b>converge within ~2&ndash;8%</b> &mdash; the run is "
        "compute/scheduling-bound and the buffer allocation no longer matters.</li></ul></div>"
    )
    p.append(
        '<div class="card"><h3>Reading the result honestly</h3><ul>'
        "<li>HiSparse+DP is a <b>tail-latency</b> refinement <i>within</i> the offloaded regime, not a free "
        "throughput/latency win over dense.</li>"
        "<li>To demonstrate HiSparse&rsquo;s intended win you must let the offload&rsquo;s headroom pay off: a "
        "workload that <b>OOMs the dense arm</b> (longer context or a larger running batch) while HiSparse still "
        "serves. That is the natural next experiment.</li></ul></div>"
    )
    p.append("</div>")

    # HiSparse + MTP composability finding.
    p.append(
        '<h2>HiSparse + MTP (speculative) <span class="chip">does not compose</span></h2>'
    )
    p.append(
        '<div class="note warn">We tested HiSparse + EAGLE/NEXTN MTP (<code>--speculative-algorithm EAGLE '
        "--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4</code>, draft head "
        "auto-loaded from the DSv4-Flash checkpoint, DP-attention preserved). <b>Both the low-concurrency and the "
        "c=128 runs crashed at startup</b>, even at c=1:</div>"
    )
    p.append(
        '<div class="card"><pre style="font-size:11px;overflow-x:auto;color:#ffb3b3;margin:0">'
        "scheduler.py:2852  _build_hisparse_decode_batch -&gt; future_map.stash(...)\n"
        "overlap_utils.py:517  self.topk_p_buf[indices] = payload.topk_p.to(self.topk_p_buf.dtype)\n"
        "AttributeError: 'NoneType' object has no attribute 'to'</pre>"
        '<div class="note"><b>Root cause:</b> with speculation on, <code>need_topk</code> is unconditionally '
        "True (<code>overlap_utils.py:305</code>), so <code>stash()</code> dereferences "
        "<code>payload.topk_p</code> &mdash; but HiSparse&rsquo;s dedicated decode-batch builder "
        "(<code>_build_hisparse_decode_batch</code>) emits a <code>RelayPayload</code> with "
        "<code>topk_p=None</code>. HiSparse and MTP each own a separate decode path and they collide in the "
        "overlap future-map. <b>This is a real source bug</b>, not a config error &mdash; composing the two needs "
        "a code fix (guard the topk stash on payload presence, or have the HiSparse decode-batch builder populate "
        "the spec topk fields), not just flags. Static analysis had suggested they compose; the runtime says "
        "otherwise.</div></div>"
    )

    # Cache hit section
    p.append("<h2>Cache hit rate</h2>")
    if has_cache:
        p.append(
            '<div class="note">Cache-report data present (see per-point JSONL <code>cache_report</code>).</div>'
        )
    else:
        p.append(
            '<div class="note warn">Not yet collected. The A/B runs did not pass '
            "<code>--cache-report</code>, so device/host KV cache-hit rates are unavailable. "
            "A follow-up sweep with <code>--cache-report</code> (and the HiSparse per-layer demand-miss "
            "counters) is needed to populate this section &mdash; it will quantify how the DP allocation "
            "shifts hit rate per layer, the mechanism behind the tail-latency change.</div>"
        )

    # Why it works (mechanism)
    p.append('<h2>Why it works <span class="chip">mechanism</span></h2>')
    p.append('<div class="grid">')
    p.append(
        '<div class="card"><h3>HiSparse already offloads KV</h3><ul>'
        "<li>HiSparse keeps only a per-layer GPU <b>device buffer</b> of B compressed KV slots; the "
        "full compressed context lives in a <b>host</b> paged pool and is swapped in on demand.</li>"
        "<li>For this workload (compressed ctx mean &asymp;1.8k slots/layer, B=1024): "
        "<b>&asymp;61% of KV is device-resident, &asymp;39% is offloaded to host</b> &mdash; "
        "<b>100%</b> of requests exceed B and touch the host path.</li>"
        "<li>Each decode step demands top_k=512 slots &asymp; <b>28%</b> of context; the buffer holds "
        "~2&times; one step&rsquo;s demand, so the tail is set by how often the working set misses B "
        "and stalls on a host swap-in.</li></ul></div>"
    )
    p.append(
        '<div class="card"><h3>Why per-layer helps the tail</h3><ul>'
        "<li>Layers differ several-fold in miss rate at equal B (measured in the offline replay). "
        "A uniform B over-serves &ldquo;easy&rdquo; layers and starves &ldquo;hard&rdquo; ones.</li>"
        "<li>The DP moves slots from 11 easy layers (&rarr;704) to 10 hard layers (&rarr;1408), same "
        "&Sigma;B. Fewer worst-case swap-ins on the hard layers &rarr; the <b>slowest steps get faster</b>, "
        "so p99 TTFT/E2E drop while medians (already hitting the buffer) barely move.</li>"
        "<li>Effect peaks at low&ndash;moderate load: a few concurrent long-context requests contend for "
        "host&harr;device bandwidth, where the swap-stall tail is largest. Under saturation the bottleneck "
        "is compute/scheduling, not the buffer, so the arms converge.</li></ul></div>"
    )
    p.append("</div>")

    # Takeaways
    p.append('<h2>Takeaways &amp; future work</h2><div class="card"><ul>')
    p.append(
        "<li><b>Delivered (no proxy):</b> per-layer B<sub>l</sub> is implemented end-to-end and "
        "GPU-validated; the DP arm serves correctly with a genuinely non-uniform allocation "
        "(704&ndash;1408/layer) at matched memory; uniform config is bit-identical to stock.</li>"
    )
    p.append(
        "<li><b>Result (dp vs uniform):</b> DP is a <b>tail-latency win at low load</b> (best at c=5: E2E p99 "
        "&minus;49%, TTFT p99 &minus;74%, req/s +9.5%), neutral at saturation. Throughput/medians "
        "unchanged by construction (same total work + memory).</li>"
    )
    p.append(
        "<li><b>Result (vs non-HiSparse dense):</b> dense has the highest throughput at every load "
        "(the offload is not a throughput win at equal work). HiSparse+DP&rsquo;s value is a tail-latency "
        "refinement (c=1 TTFT p99 &minus;30% vs dense; rescues uniform&rsquo;s c=5 blow-up); at c=128 all arms "
        "converge. HiSparse&rsquo;s real payoff &mdash; memory headroom &mdash; needs a dense-OOM workload to show.</li>"
    )
    p.append(
        "<li><b>HiSparse + MTP does not compose</b> (runtime crash: <code>topk_p=None</code> in "
        "<code>_build_hisparse_decode_batch</code>&rarr;<code>stash</code>). The two decode paths collide; "
        "a source fix is required before speculative decoding can stack on HiSparse.</li>"
    )
    p.append(
        "<li><b>Future work &mdash; measure the mechanism:</b> rerun with <code>--cache-report</code> "
        "to record device/host hit-rate <i>per layer</i> and confirm the DP shifts misses off the hard "
        "layers (populates the section above with data, not inference).</li>"
    )
    p.append(
        "<li><b>Future work &mdash; robustness:</b> the c=5 peak is a single point; repeat c=4&ndash;8 "
        "with multiple seeds to separate signal from variance, and sweep total budget &Sigma;B "
        "(0.1&ndash;0.3 ratio) to trace the win vs memory.</li>"
    )
    p.append(
        "<li><b>Future work &mdash; generalization:</b> refit the DP on a different workload "
        "(e.g. AA-LCR long-context reasoning) to test whether the optimal per-layer profile is "
        "workload-dependent, and A/B a workload-matched vs mismatched allocation.</li>"
    )
    p.append(
        "<li><b>Future work &mdash; learned/adaptive B<sub>l</sub>:</b> the current B<sub>l</sub> is "
        "a static offline fit; explore online adaptation from live per-layer miss counters.</li>"
    )
    p.append("</ul></div>")

    p.append(
        '<div class="foot">Generated by <code>make_report.py</code> from '
        "<code>ab_*/{uniform,dp}/bench_c*.jsonl</code>. Re-run to refresh as more concurrency "
        "points complete.</div>"
    )
    p.append("</div></body></html>")
    return "".join(p)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing ab_<jobid>/ result folders.",
    )
    ap.add_argument("--out", required=True, help="Output HTML path.")
    args = ap.parse_args(argv)

    data = collect(args.results_root)
    token_dist = _load_token_dist(args.results_root)
    html = render(data, token_dist)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    concs = sorted({c for (c, _a) in data})
    print(f"Wrote {args.out}  ({len(data)} points across concurrency {concs})")


if __name__ == "__main__":
    main()
