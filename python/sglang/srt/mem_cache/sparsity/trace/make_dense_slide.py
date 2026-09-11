"""Generate a 16:9 NVIDIA-themed slide for the DENSE (no-HiSparse) baseline.

Companion to ``make_slide.py`` (the uniform-vs-dp per-layer study). This slide
frames the third arm: DSv4-Flash running its NATIVE compressed + C4Indexer
sparse attention but WITHOUT HiSparse's host offload, so the full compressed KV
stays on GPU (pool_configurator c4_shrink_factor=1). It is the "no-offload
reference": it shows what HiSparse's offload costs (or saves) at matched work.

Reads:
  * dense arm   from ``dense_*/dense/bench_c*.jsonl``
  * uniform/dp  from ``ab_*/{uniform,dp}/bench_c*.jsonl``
Focuses on LOW concurrency (1-10), where the host<->device swap contention that
HiSparse introduces is most visible. Run by file path (no sglang import).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

ARMS = ("dense", "uniform", "dp")


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


def collect(
    results_root: str, job_glob: Optional[str] = None
) -> Dict[Tuple[int, str], dict]:
    """(concurrency, arm) -> latest bench dict, across all job dirs.

    dense       lives in dense_* (low conc) and hc_* (high conc);
    uniform/dp  live in ab_* (both ranges) and hc_* (high conc).
    Later job ids override earlier ones for the same (conc, arm).

    ``job_glob`` overrides the default multi-source scan with a single glob
    (e.g. ``swelen_*``) whose subdirs hold all three arms -- used for a
    self-contained run (like the SWE-bench realistic-output sweep) that should
    NOT be mixed with the fixed-256 A/B data.
    """
    out: Dict[Tuple[int, str], dict] = {}
    if job_glob is not None:
        patterns = [(os.path.join(results_root, job_glob), ("dense", "uniform", "dp"))]
    else:
        patterns = [
            (os.path.join(results_root, "dense_*"), ("dense",)),
            (os.path.join(results_root, "ab_*"), ("uniform", "dp")),
            (os.path.join(results_root, "hc_*"), ("dense", "uniform", "dp")),
        ]
    for root_glob, arms in patterns:
        for job_dir in sorted(glob.glob(root_glob)):
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


def _delta(a, b):
    """Percent change of a relative to b."""
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def _cls(dl, lower_is_better=True):
    if dl is None or abs(dl) < 2.0:
        return "neu"
    good = (dl < 0) if lower_is_better else (dl > 0)
    return "good" if good else "bad"


def _d(dl):
    return "&ndash;" if dl is None else f"{dl:+.0f}%"


def _num(v, fmt="{:.0f}"):
    return "&ndash;" if v is None else fmt.format(v)


SLIDE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>HiSparse vs Dense Baseline &mdash; Slide</title>
<style>
  :root{{--nv:#76b900;--ink:#0b0e11;--panel:#161a20;--line:#2c333d;--txt:#eef2f6;--muted:#9aa4b2;--good:#76b900;--bad:#ff5c5c;--neu:#8892a0;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  html,body{{background:#000;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;}}
  .slide{{width:1280px;height:720px;margin:12px auto;background:linear-gradient(135deg,#0b0e11,#12161c);
    color:var(--txt);position:relative;overflow:hidden;border:1px solid var(--line);}}
  .bar{{position:absolute;left:0;top:0;width:8px;height:100%;background:var(--nv);}}
  .hd{{padding:24px 40px 6px 48px;}}
  .hd h1{{font-size:28px;font-weight:800;letter-spacing:.2px;}}
  .hd h1 .g{{color:var(--nv);}}
  .hd .sub{{color:var(--muted);font-size:13.5px;margin-top:4px;}}
  .body{{display:grid;grid-template-columns:1.04fr 1fr;gap:18px;padding:8px 40px 18px 48px;height:calc(100% - 84px);}}
  .col{{display:flex;flex-direction:column;gap:11px;}}
  .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 18px;}}
  .card h3{{color:var(--nv);font-size:11.5px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:7px;}}
  .card li{{font-size:12.5px;margin:3px 0 3px 16px;}}
  code{{background:#0a0d10;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11.5px;color:#cfe8a8;}}
  table{{border-collapse:collapse;width:100%;font-size:12px;}}
  th,td{{padding:4px 5px;text-align:right;border-bottom:1px solid var(--line);}}
  th{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.3px;}}
  td.l,th.l{{text-align:left;}}
  .good{{color:var(--good);font-weight:700;}} .bad{{color:var(--bad);font-weight:700;}} .neu{{color:var(--neu);}}
  .base{{color:#e0b050;font-weight:700;}}
  .pill{{display:inline-block;font-size:10px;font-weight:700;padding:1px 7px;border-radius:8px;}}
  .p-de{{background:#3a2a10;color:#e0b050;}} .p-un{{background:#1d2937;color:#8fb3ff;}}
  .p-dp{{background:#25350a;color:var(--nv);}} .p-mtp{{background:#3a1030;color:#ff8fd0;}}
  .kpis{{display:flex;gap:10px;}}
  .kpi{{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:11px;text-align:center;}}
  .kpi .n{{font-size:22px;font-weight:800;color:var(--nv);}}
  .kpi .l{{font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;margin-top:2px;}}
  .foot{{position:absolute;bottom:12px;left:48px;right:40px;display:flex;justify-content:space-between;
    color:var(--muted);font-size:11px;border-top:1px solid var(--line);padding-top:8px;}}
  .note{{font-size:10.5px;color:var(--muted);margin-top:5px;}}
</style></head><body><div class="slide"><div class="bar"></div>
<div class="hd">
  <h1>HiSparse vs <span class="g">non-HiSparse (dense) baseline</span> &mdash; all numbers relative to dense</h1>
  <div class="sub">{subtitle}</div>
</div>
<div class="body">
  <div class="col">
    <div class="card"><h3>Baseline &amp; arms</h3><ul>
      <li><span class="pill p-de">dense</span> <b>reference (no <code>--enable-hisparse</code>).</b> Native
          compressed + C4Indexer top-k attention; full compressed KV on GPU (<code>c4_shrink_factor=1</code>).
          Fits at the <b>same 262k-token</b> budget &mdash; no OOM.</li>
      <li><span class="pill p-un">uniform</span> HiSparse, 1024 slots/layer.
          <span class="pill p-dp">dp</span> HiSparse, DP per-layer 704&ndash;1408 (same &Sigma;B).</li>
      <li><span class="pill p-mtp">dp+MTP</span> HiSparse + EAGLE/NEXTN speculative:
          <b class="bad">does not run</b> &mdash; crashes in <code>_build_hisparse_decode_batch</code>
          (<code>topk_p=None</code>). The two decode paths collide; needs a source fix.</li>
    </ul></div>
    <div class="card"><h3>What the deltas say</h3><ul>
      <li><b>Throughput:</b> dense is the <b>highest</b> at every load &mdash; HiSparse's offload is not a
          throughput win at equal work; its value is <b>memory headroom</b> this matched test doesn't reward.</li>
      <li><b>Low load (1&ndash;5):</b> the offload adds ~9&ndash;11% E2E; <b>dp claws it back</b> and beats dense
          on TTFT p99 at c=1; <b>dp rescues uniform's c=5 tail blow-up.</b></li>
      <li><b>High load (128):</b> all arms <b>converge</b> (within ~2%) &mdash; compute/scheduling-bound,
          buffer shape no longer matters.</li>
    </ul>
    <div class="note">To show a HiSparse <i>win</i> you must let the offload's headroom pay off &mdash; a workload
      that OOMs dense (longer context / larger running batch) than HiSparse can still serve.</div>
    </div>
  </div>
  <div class="col">
    <div class="kpis">{kpis}</div>
    <div class="card"><h3>&Delta; vs dense &mdash; request throughput (higher = better)</h3>
      {tput_table}
      <div class="note"><span class="good">green = faster than dense</span>,
        <span class="bad">red = slower than dense</span>. &Delta; = (arm&minus;dense)/dense.</div>
    </div>
    <div class="card"><h3>&Delta; vs dense &mdash; E2E p99 &amp; TTFT p99 (lower = better)</h3>
      {lat_table}
      <div class="note">Positive = HiSparse slower than dense (offload cost); negative = faster despite offload.
        Absolute dense p99 E2E (ms): {abs_dense}.</div>
    </div>
  </div>
</div>
<div class="foot"><span>NVIDIA Inference &middot; HiSparse vs dense reference</span>
  <span>{npts} paired points {conc_str} &middot; auto-generated</span></div>
</div>
</body></html>"""


_DEFAULT_SUBTITLE = (
    "DeepSeek-V4-Flash &middot; sglang (TP4/DP4, 4&times;GB300) &middot; "
    "SWE-bench &middot; 300 prompts, fixed 256-tok output &middot; matched memory"
)


def render(data, subtitle=_DEFAULT_SUBTITLE):
    # Every concurrency where we have a dense reference to divide by.
    concs = sorted({c for (c, _a) in data})
    dpts = [c for c in concs if (c, "dense") in data]

    # KPIs: best dp throughput vs dense, best dp E2E p99 vs dense, high-load
    # convergence, arms compared.
    best_tput = None  # (c, delta) most favorable dp req/s vs dense
    best_e2e = None  # (c, delta) most favorable dp E2E p99 vs dense
    for c in dpts:
        de = data[(c, "dense")]
        dp = data.get((c, "dp"))
        if not dp:
            continue
        dt = _delta(dp.get("request_throughput"), de.get("request_throughput"))
        if dt is not None and (best_tput is None or dt > best_tput[1]):
            best_tput = (c, dt)
        de99 = _delta(dp.get("p99_e2e_latency_ms"), de.get("p99_e2e_latency_ms"))
        if de99 is not None and (best_e2e is None or de99 < best_e2e[1]):
            best_e2e = (c, de99)

    kpis = []
    if best_e2e:
        kpis.append(
            f'<div class="kpi"><div class="n">{best_e2e[1]:+.0f}%</div>'
            f'<div class="l">best dp E2E p99 vs dense (c={best_e2e[0]})</div></div>'
        )
    if best_tput:
        kpis.append(
            f'<div class="kpi"><div class="n">{best_tput[1]:+.0f}%</div>'
            f'<div class="l">best dp req/s vs dense (c={best_tput[0]})</div></div>'
        )
    n_arms = len({a for (_c, a) in data})
    kpis.append(
        f'<div class="kpi"><div class="n">{n_arms}</div>'
        f'<div class="l">arms vs dense</div></div>'
    )
    kpis.append(
        '<div class="kpi"><div class="n">262k</div>'
        '<div class="l">dense budget (no OOM)</div></div>'
    )

    # Throughput delta table: uniform & dp req/s relative to dense.
    tput = [
        '<table><thead><tr><th class="l">conc</th>'
        "<th>dense req/s</th><th>&Delta;un</th><th>&Delta;dp</th></tr></thead><tbody>"
    ]
    for c in dpts:
        de = data[(c, "dense")]
        un = data.get((c, "uniform"))
        dp = data.get((c, "dp"))
        du = _delta(
            un.get("request_throughput") if un else None, de.get("request_throughput")
        )
        dd = _delta(
            dp.get("request_throughput") if dp else None, de.get("request_throughput")
        )
        tput.append(
            f'<tr><td class="l">{c}</td>'
            f'<td class="base">{_num(de.get("request_throughput"), "{:.3f}")}</td>'
            f'<td class="{_cls(du, False)}">{_d(du)}</td>'
            f'<td class="{_cls(dd, False)}">{_d(dd)}</td></tr>'
        )
    tput.append("</tbody></table>")

    # Latency delta table: E2E p99 and TTFT p99 for uniform & dp vs dense.
    lat = [
        '<table><thead><tr><th class="l">conc</th>'
        "<th>&Delta;un E2E</th><th>&Delta;dp E2E</th>"
        "<th>&Delta;un TTFT</th><th>&Delta;dp TTFT</th></tr></thead><tbody>"
    ]
    for c in dpts:
        de = data[(c, "dense")]
        un = data.get((c, "uniform"))
        dp = data.get((c, "dp"))
        ue = _delta(
            un.get("p99_e2e_latency_ms") if un else None, de.get("p99_e2e_latency_ms")
        )
        pe = _delta(
            dp.get("p99_e2e_latency_ms") if dp else None, de.get("p99_e2e_latency_ms")
        )
        ut = _delta(un.get("p99_ttft_ms") if un else None, de.get("p99_ttft_ms"))
        pt = _delta(dp.get("p99_ttft_ms") if dp else None, de.get("p99_ttft_ms"))
        lat.append(
            f'<tr><td class="l">{c}</td>'
            f'<td class="{_cls(ue)}">{_d(ue)}</td>'
            f'<td class="{_cls(pe)}">{_d(pe)}</td>'
            f'<td class="{_cls(ut)}">{_d(ut)}</td>'
            f'<td class="{_cls(pt)}">{_d(pt)}</td></tr>'
        )
    lat.append("</tbody></table>")

    abs_dense = (
        ", ".join(
            f"c{c}={data[(c, 'dense')].get('p99_e2e_latency_ms', 0):.0f}" for c in dpts
        )
        or "&ndash;"
    )

    return SLIDE.format(
        subtitle=subtitle,
        kpis="".join(kpis),
        tput_table="".join(tput),
        lat_table="".join(lat),
        abs_dense=abs_dense,
        npts=len(dpts),
        conc_str="(" + ",".join(str(c) for c in dpts) + ")" if dpts else "",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-root", default=os.path.join(os.path.dirname(__file__), "results")
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--job-glob",
        default=None,
        help="Single job-dir glob (e.g. 'swelen_*') whose subdirs hold "
        "all three arms; used for a self-contained run not to be "
        "mixed with the fixed-256 A/B data.",
    )
    ap.add_argument(
        "--subtitle",
        default=None,
        help="Override the slide subtitle (HTML entities allowed).",
    )
    args = ap.parse_args(argv)
    data = collect(args.results_root, job_glob=args.job_glob)
    html = render(data, subtitle=args.subtitle or _DEFAULT_SUBTITLE)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    concs = sorted({c for (c, _a) in data if (c, "dense") in data})
    arms = sorted({a for (_c, a) in data})
    print(f"Wrote {args.out} (dense-referenced conc {concs}, arms {arms})")


if __name__ == "__main__":
    main()
