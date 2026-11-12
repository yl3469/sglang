"""Generate a single 16:9 NVIDIA-themed summary SLIDE for the HiSparse A/B study.

Companion to ``make_report.py`` (which produces the detailed multi-section
report). This renders ONE printable 16:9 slide: title, the core design choice,
experimental setup, and a compact headline results panel (the DP tail-latency
win vs load). Reads the same ``ab_*/{uniform,dp}/bench_c*.jsonl`` data. Run by
file path (no sglang import needed); re-run to refresh as points land.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from report_assets import kernel_shapes_svg
except Exception:  # pragma: no cover
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
    """(conc, arm) -> latest bench dict. uniform/dp from ab_* and hc_*; the
    non-HiSparse ``dense`` reference from dense_* and hc_*."""
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


def collect_swelen(results_root: str) -> Dict[Tuple[int, str], dict]:
    """Realistic gold-patch-output runs live in swelen_* dirs, each holding all
    three arms (dense/uniform/dp). Kept separate from the fixed-256 A/B data."""
    out: Dict[Tuple[int, str], dict] = {}
    for job_dir in sorted(glob.glob(os.path.join(results_root, "swelen_*"))):
        for arm in ("dense", "uniform", "dp"):
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


def _delta(dp, un):
    if dp is None or un is None or un == 0:
        return None
    return (dp - un) / un * 100.0


def _cls(dl, lower_is_better=True):
    if dl is None or abs(dl) < 2.0:
        return "neu"
    good = (dl < 0) if lower_is_better else (dl > 0)
    return "good" if good else "bad"


def _d(dl):
    return "&ndash;" if dl is None else f"{dl:+.0f}%"


SLIDE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>HiSparse Per-Layer Buffer &mdash; Slide</title>
<style>
  :root{{--nv:#76b900;--ink:#0b0e11;--panel:#161a20;--line:#2c333d;--txt:#eef2f6;--muted:#9aa4b2;--good:#76b900;--bad:#ff5c5c;--neu:#8892a0;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  html,body{{background:#000;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;}}
  .slide{{width:1280px;height:720px;margin:12px auto;background:linear-gradient(135deg,#0b0e11,#12161c);
    color:var(--txt);position:relative;overflow:hidden;border:1px solid var(--line);}}
  .bar{{position:absolute;left:0;top:0;width:8px;height:100%;background:var(--nv);}}
  .hd{{padding:26px 40px 10px 48px;}}
  .hd h1{{font-size:30px;font-weight:800;letter-spacing:.2px;}}
  .hd h1 .g{{color:var(--nv);}}
  .hd .sub{{color:var(--muted);font-size:14px;margin-top:4px;}}
  .body{{display:grid;grid-template-columns:1.05fr 1fr;gap:20px;padding:8px 40px 20px 48px;height:calc(100% - 96px);}}
  .col{{display:flex;flex-direction:column;gap:14px;}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:11px 18px;}}
  .card h3{{color:var(--nv);font-size:12px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:8px;}}
  .card li{{font-size:13.5px;margin:4px 0 4px 16px;}}
  code{{background:#0a0d10;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:12px;color:#cfe8a8;}}
  table{{border-collapse:collapse;width:100%;font-size:13px;}}
  th,td{{padding:3px 6px;text-align:right;border-bottom:1px solid var(--line);}}
  th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.3px;}}
  td.l,th.l{{text-align:left;}}
  .good{{color:var(--good);font-weight:700;}} .bad{{color:var(--bad);font-weight:700;}} .neu{{color:var(--neu);}}
  .kpis{{display:flex;gap:12px;}}
  .kpi{{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;text-align:center;}}
  .kpi .n{{font-size:26px;font-weight:800;color:var(--nv);}}
  .kpi .l{{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;margin-top:2px;}}
  .foot{{position:absolute;bottom:12px;left:48px;right:40px;display:flex;justify-content:space-between;
    color:var(--muted);font-size:11px;border-top:1px solid var(--line);padding-top:8px;}}
  .note{{font-size:11px;color:var(--muted);margin-top:4px;}}
</style></head><body><div class="slide"><div class="bar"></div>
<div class="hd">
  <h1>Per-Layer HiSparse Device Buffer: <span class="g">DP &gt; Uniform</span> at equal memory</h1>
  <div class="sub">DeepSeek-V4-Flash &middot; sglang HiSparse (TP4/DP4, 4&times;GB300) &middot; SWE-bench &middot; matched-memory A/B serving sweep</div>
</div>
<div class="body">
  <div class="col">
    <div class="card"><h3>Key design choice</h3><ul>
      <li>Replace HiSparse's single scalar <code>device_buffer_size</code> with a
          <b>per-layer</b> capacity B<sub>l</sub> over the 21 C4A indexer layers.</li>
      <li>B<sub>l</sub> comes from the offline <b>DP allocation</b> (trace &rarr; LRU miss curves
          &rarr; min-plus DP), fed via <code>--hisparse-config device_buffer_sizes</code>.</li>
      <li>Contained: 2 hot-path lines (per-layer kernel <code>hot_buffer_size</code> +
          reserved-slot scatter); physical tensors stay at MAX(B<sub>l</sub>).</li>
      <li>Uniform config is <b>bit-identical to stock</b> (proven) &rarr; trustworthy baseline.</li>
    </ul></div>
    <div class="card"><h3>Experimental setup</h3><ul>
      <li><b>Baseline (uniform):</b> 1024 slots/layer. <b>Treatment (dp):</b> 704&ndash;1408/layer,
          <b>same total</b> &Sigma;B<sub>l</sub> &rarr; isolates allocation shape.</li>
      <li>200 req/point, <b>fixed 256-tok output</b> (ignore_eos); metrics TTFT / TBT / E2E.
          Concurrency sweep: {concs}.</li>
    </ul></div>
    <div class="card"><h3>vs no-HiSparse baseline &mdash; &Delta; req/s &amp; E2E p99</h3>
      {dense_table}
      <div class="note"><b>dense</b> = no <code>--enable-hisparse</code> (full KV on GPU, no host offload;
        fits at 262k tokens). &Delta; = (arm&minus;dense)/dense: <span class="good">green = HiSparse better</span>,
        <span class="bad">red = offload costs</span>. Dense wins throughput at equal work; HiSparse's value is memory headroom.</div>
    </div>
  </div>
  <div class="col">
    <div class="kpis">{kpis}</div>
    <div class="card"><h3>Headline results &mdash; &Delta;dp vs uniform (E2E latency)</h3>
      {table}
      <div class="note">&Delta;dp = (dp&minus;uniform)/uniform. <span class="good">green = DP faster / higher</span>,
        <span class="bad">red = worse</span>. Throughput &amp; medians unchanged (same work);
        DP shifts buffer to needy layers &rarr; fewer swap stalls &rarr; lower tails.</div>
    </div>
    <div class="note">Cache-hit rate: pending a <code>--cache-report</code> rerun (not yet collected).</div>
  </div>
</div>
<div class="foot"><span>NVIDIA Inference &middot; HiSparse per-layer capacity</span>
  <span>{npairs} paired concurrency points &middot; auto-generated</span></div>
</div>
{slide2}
</body></html>"""

SLIDE2 = """<div class="slide"><div class="bar"></div>
<div class="hd">
  <h1>Touched kernel path &mdash; <span class="g">tensor shapes</span></h1>
  <div class="sub">Per-layer B<sub>l</sub> changes only the kernel template &amp; the reserved-slot column; no tensor is reshaped</div>
</div>
<div style="padding:6px 40px 12px 48px;">
  <div class="card">{diagram}</div>
  <div class="body" style="height:auto;padding:0;margin-top:12px;">
    <div class="col"><div class="card"><h3>Why it works</h3><ul>
      <li>HiSparse offloads KV: only B compressed slots/layer stay on GPU, rest on host, swapped in on demand.
          Here <b>&asymp;61% device-resident, &asymp;39% host-offloaded</b> (ctx&asymp;1.8k, B=1024); every request hits host.</li>
      <li>Layers differ several-fold in miss rate at equal B. DP moves slots from 11 easy layers (&rarr;704)
          to 10 hard ones (&rarr;1408) &rarr; fewer worst-case swap-ins &rarr; <b>lower p99</b>; medians unchanged.</li>
      <li>Peaks at low load (host&harr;device contention); converges at saturation (compute-bound).</li>
    </ul></div></div>
    <div class="col"><div class="card"><h3>Takeaways &amp; future work</h3><ul>
      <li><b>Delivered, no proxy:</b> per-layer B<sub>l</sub> end-to-end + GPU-validated; uniform == stock.</li>
      <li><b>Result:</b> tail-latency win at low load (best c=5: E2E p99 &minus;49%, TTFT p99 &minus;74%, +9.5% req/s); neutral at saturation.</li>
      <li><b>Next:</b> <code>--cache-report</code> for per-layer hit-rate (measure the mechanism);
          repeat c=4&ndash;8 &times; seeds; sweep &Sigma;B budget.</li>
      <li><b>Next:</b> refit DP on other workloads (AA-LCR); explore learned/online B<sub>l</sub>.</li>
    </ul></div></div>
  </div>
</div>
<div class="foot"><span>NVIDIA Inference &middot; HiSparse per-layer capacity</span>
  <span>implementation design &middot; auto-generated</span></div>
</div>"""

# Slide 3: the realistic SWE-bench output-length profile (per-request output
# follows the gold-patch length, mean ~598 tok) vs the fixed-256 A/B above.
SLIDE3 = """<div class="slide"><div class="bar"></div>
<div class="hd">
  <h1>Realistic output length &mdash; <span class="g">SWE-bench gold-patch profile</span></h1>
  <div class="sub">Per-request output follows the real gold-patch length (mean &asymp;598 tok, long tail) instead of a flat 256 &middot; all &Delta; vs no-HiSparse dense</div>
</div>
<div class="body">
  <div class="col">
    <div class="card"><h3>What changes vs the fixed-256 A/B</h3><ul>
      <li>Output length = tokenized gold patch per request (loader derives it when
          <code>--sharegpt-output-len</code> is omitted). <b>Decode-heavy</b>: mean &asymp;598 tok, tail to 2000+.</li>
      <li>Deterministic per request &rarr; identical work across arms, so the A/B stays clean;
          only the length <i>profile</i> becomes realistic.</li>
      <li><b>DP&asymp;uniform now:</b> DP helps the prefill/TTFT tail, but E2E is now decode-dominated,
          so the shape win is diluted; and variable lengths desync requests, removing uniform's c=5 cliff.</li>
    </ul></div>
    <div class="card"><h3>Setup</h3><ul>
      <li>Same server/model/memory as the fixed-256 sweep; 100 prompts/point.</li>
      <li>Concurrency: {sw_concs}. Arms: dense (no HiSparse), uniform, dp.</li>
    </ul></div>
  </div>
  <div class="col">
    <div class="card"><h3>&Delta; vs dense &mdash; req/s &amp; E2E p99</h3>
      {sw_dense_table}
      <div class="note"><span class="good">green = HiSparse better than dense</span>,
        <span class="bad">red = worse</span>. Dense leads on throughput &amp; E2E across the board;
        HiSparse's edge is TTFT p99.</div>
    </div>
    <div class="card"><h3>DP vs uniform &mdash; &Delta;E2E p99 / &Delta;TTFT p99</h3>
      {sw_dpun_table}
      <div class="note">Compare to the fixed-256 slide (c=5 was E2E p99 &minus;49%): with real outputs the
        dp&ndash;uniform gap collapses to &asymp;0 &mdash; DP's prefill-tail win is diluted by long decode.</div>
    </div>
  </div>
</div>
<div class="foot"><span>NVIDIA Inference &middot; HiSparse per-layer capacity</span>
  <span>realistic output-length study &middot; auto-generated</span></div>
</div>"""


def _load_dp_sizes(results_root):
    p = os.path.join(results_root, "_shared", "token_dist.json")
    if os.path.exists(p):
        try:
            return json.load(open(p)).get("dp_buffer_sizes")
        except Exception:
            return None
    return None


def render(data, dp_sizes=None, swelen=None):
    concs = sorted({c for (c, _a) in data})
    pairs = [c for c in concs if (c, "uniform") in data and (c, "dp") in data]

    # KPIs: best E2E p99 win, best req/s win, #points.
    best_e2e = None
    best_req = None
    for c in pairs:
        u, d = data[(c, "uniform")], data[(c, "dp")]
        de = _delta(d.get("p99_e2e_latency_ms"), u.get("p99_e2e_latency_ms"))
        if de is not None and (best_e2e is None or de < best_e2e[1]):
            best_e2e = (c, de)
        dr = _delta(d.get("request_throughput"), u.get("request_throughput"))
        if dr is not None and (best_req is None or dr > best_req[1]):
            best_req = (c, dr)

    kpis = []
    if best_e2e:
        kpis.append(
            f'<div class="kpi"><div class="n">{best_e2e[1]:+.0f}%</div>'
            f'<div class="l">best E2E p99 (c={best_e2e[0]})</div></div>'
        )
    if best_req:
        kpis.append(
            f'<div class="kpi"><div class="n">{best_req[1]:+.0f}%</div>'
            f'<div class="l">best req/s (c={best_req[0]})</div></div>'
        )
    kpis.append(
        f'<div class="kpi"><div class="n">{len(pairs)}</div>'
        f'<div class="l">load points</div></div>'
    )
    kpis.append(
        '<div class="kpi"><div class="n">=mem</div>'
        '<div class="l">matched budget</div></div>'
    )

    # Compact table: per concurrency, E2E p50/p90/p99 deltas + req/s delta.
    rows = [
        '<table><thead><tr><th class="l">conc</th><th>&Delta;req/s</th>'
        "<th>&Delta;E2E p50</th><th>&Delta;E2E p90</th><th>&Delta;E2E p99</th>"
        "<th>&Delta;TTFT p99</th></tr></thead><tbody>"
    ]
    for c in pairs:
        u, d = data[(c, "uniform")], data[(c, "dp")]
        dr = _delta(d.get("request_throughput"), u.get("request_throughput"))
        e50 = _delta(d.get("median_e2e_latency_ms"), u.get("median_e2e_latency_ms"))
        e90 = _delta(d.get("p90_e2e_latency_ms"), u.get("p90_e2e_latency_ms"))
        e99 = _delta(d.get("p99_e2e_latency_ms"), u.get("p99_e2e_latency_ms"))
        t99 = _delta(d.get("p99_ttft_ms"), u.get("p99_ttft_ms"))
        rows.append(
            f'<tr><td class="l">{c}</td>'
            f'<td class="{_cls(dr, False)}">{_d(dr)}</td>'
            f'<td class="{_cls(e50)}">{_d(e50)}</td>'
            f'<td class="{_cls(e90)}">{_d(e90)}</td>'
            f'<td class="{_cls(e99)}">{_d(e99)}</td>'
            f'<td class="{_cls(t99)}">{_d(t99)}</td></tr>'
        )
    rows.append("</tbody></table>")

    # Dense-baseline table: uniform & dp vs the non-HiSparse dense reference.
    dense_concs = [c for c in concs if (c, "dense") in data]
    if dense_concs:
        drows = [
            '<table><thead><tr><th class="l">conc</th>'
            "<th>&Delta;un req/s</th><th>&Delta;dp req/s</th>"
            "<th>&Delta;un E2E p99</th><th>&Delta;dp E2E p99</th></tr></thead><tbody>"
        ]
        for c in dense_concs:
            de = data[(c, "dense")]
            u = data.get((c, "uniform"))
            d = data.get((c, "dp"))
            ur = (
                _delta(u.get("request_throughput"), de.get("request_throughput"))
                if u
                else None
            )
            dr = (
                _delta(d.get("request_throughput"), de.get("request_throughput"))
                if d
                else None
            )
            ue = (
                _delta(u.get("p99_e2e_latency_ms"), de.get("p99_e2e_latency_ms"))
                if u
                else None
            )
            deb = (
                _delta(d.get("p99_e2e_latency_ms"), de.get("p99_e2e_latency_ms"))
                if d
                else None
            )
            drows.append(
                f'<tr><td class="l">{c}</td>'
                f'<td class="{_cls(ur, False)}">{_d(ur)}</td>'
                f'<td class="{_cls(dr, False)}">{_d(dr)}</td>'
                f'<td class="{_cls(ue)}">{_d(ue)}</td>'
                f'<td class="{_cls(deb)}">{_d(deb)}</td></tr>'
            )
        drows.append("</tbody></table>")
        dense_table = "".join(drows)
    else:
        dense_table = (
            '<div class="note">No dense (no-HiSparse) baseline points found '
            "(run <code>dense_baseline.sbatch</code>).</div>"
        )

    slide2 = ""
    if kernel_shapes_svg is not None:
        diagram = kernel_shapes_svg(
            layer_num=21,
            top_k=512,
            b_uniform=1024,
            b_dp=dp_sizes,
            page_size=256,
            width=1160,
            compact=True,
        )
        slide2 = SLIDE2.format(diagram=diagram)

    slide3 = _render_swelen_slide(swelen) if swelen else ""

    return (
        SLIDE.format(
            concs=", ".join(str(c) for c in concs) if concs else "(pending)",
            kpis="".join(kpis),
            table="".join(rows),
            dense_table=dense_table,
            npairs=len(pairs),
            slide2=slide2,
        )
        + slide3
    )


def _render_swelen_slide(sw):
    """Build slide 3 from the realistic gold-patch-output run (swelen_*)."""
    sw_concs = sorted({c for (c, _a) in sw})
    dense_concs = [c for c in sw_concs if (c, "dense") in sw]

    # vs-dense table (all arms relative to dense).
    dt = [
        '<table><thead><tr><th class="l">conc</th>'
        "<th>&Delta;un req/s</th><th>&Delta;dp req/s</th>"
        "<th>&Delta;un E2E p99</th><th>&Delta;dp E2E p99</th></tr></thead><tbody>"
    ]
    for c in dense_concs:
        de = sw[(c, "dense")]
        u = sw.get((c, "uniform"))
        d = sw.get((c, "dp"))
        ur = (
            _delta(u.get("request_throughput"), de.get("request_throughput"))
            if u
            else None
        )
        dr = (
            _delta(d.get("request_throughput"), de.get("request_throughput"))
            if d
            else None
        )
        ue = (
            _delta(u.get("p99_e2e_latency_ms"), de.get("p99_e2e_latency_ms"))
            if u
            else None
        )
        dd = (
            _delta(d.get("p99_e2e_latency_ms"), de.get("p99_e2e_latency_ms"))
            if d
            else None
        )
        dt.append(
            f'<tr><td class="l">{c}</td>'
            f'<td class="{_cls(ur, False)}">{_d(ur)}</td>'
            f'<td class="{_cls(dr, False)}">{_d(dr)}</td>'
            f'<td class="{_cls(ue)}">{_d(ue)}</td>'
            f'<td class="{_cls(dd)}">{_d(dd)}</td></tr>'
        )
    dt.append("</tbody></table>")

    # dp-vs-uniform table.
    pt = [
        '<table><thead><tr><th class="l">conc</th>'
        "<th>&Delta;dp E2E p99</th><th>&Delta;dp TTFT p99</th></tr></thead><tbody>"
    ]
    for c in sw_concs:
        u = sw.get((c, "uniform"))
        d = sw.get((c, "dp"))
        if not (u and d):
            continue
        e99 = _delta(d.get("p99_e2e_latency_ms"), u.get("p99_e2e_latency_ms"))
        t99 = _delta(d.get("p99_ttft_ms"), u.get("p99_ttft_ms"))
        pt.append(
            f'<tr><td class="l">{c}</td>'
            f'<td class="{_cls(e99)}">{_d(e99)}</td>'
            f'<td class="{_cls(t99)}">{_d(t99)}</td></tr>'
        )
    pt.append("</tbody></table>")

    return SLIDE3.format(
        sw_concs=", ".join(str(c) for c in sw_concs) if sw_concs else "(pending)",
        sw_dense_table="".join(dt),
        sw_dpun_table="".join(pt),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root", default=os.path.join(os.path.dirname(__file__), "results")
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data = collect(args.results_root)
    dp_sizes = _load_dp_sizes(args.results_root)
    swelen = collect_swelen(args.results_root)
    html = render(data, dp_sizes, swelen=swelen)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    concs = sorted({c for (c, _a) in data})
    print(f"Wrote {args.out} (concurrency {concs})")


if __name__ == "__main__":
    main()
