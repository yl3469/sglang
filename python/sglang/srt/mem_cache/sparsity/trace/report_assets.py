"""Shared inline-SVG assets for the HiSparse A/B report and slide.

Currently: a "tensor shapes on the touched kernel path" diagram that visualises
which HiSparse buffers change (and which do NOT) when the single scalar
``device_buffer_size`` becomes a per-layer vector B_l. Pure string builders, no
dependencies, so both ``make_report.py`` and ``make_slide.py`` (run by file
path) can import it.

Geometry facts encoded (from the coordinator + kernel study):
* One HiSparseCoordinator per model; L = 21 C4A indexer layers for DSv4-Flash.
* Physical buffers keep width = MAX(B_l) + page_size (unchanged shapes).
* Per-layer capacity B_l is a *logical* cap: it only drives the kernel template
  ``hot_buffer_size`` and the reserved-slot column (offset B_l per layer).
* ``req_to_device_buffer`` has NO layer axis; the other three carry a leading
  layer dim already.
"""

from __future__ import annotations

from typing import Optional, Sequence


def _bar(x, y, w, h, fill, stroke="#2c333d", rx=3):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
    )


def _txt(x, y, s, fill="#e7ebf0", size=11, anchor="start", weight="normal", mono=False):
    fam = "ui-monospace,Menlo,Consolas,monospace" if mono else "inherit"
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-family="{fam}">{s}</text>'
    )


def kernel_shapes_svg(
    layer_num: int = 21,
    top_k: int = 512,
    b_uniform: int = 1024,
    b_dp: Optional[Sequence[int]] = None,
    page_size: int = 256,
    width: int = 900,
    compact: bool = False,
) -> str:
    """Return an inline SVG of the touched-kernel-path tensor shapes.

    ``compact`` trims labels/height for the slide.
    """
    if b_dp is None:
        b_dp = (
            [704] * 6
            + [1344] * 5
            + [1408, 1408, 704, 1408, 1408, 704, 704, 1408, 704, 704]
        )
    max_b = max([b_uniform] + list(b_dp))
    padded = max_b + page_size

    NV = "#76b900"
    AMBER = "#c9a227"
    GREY = "#3a4250"
    PANEL = "#161a20"
    row_h = 40 if not compact else 32
    gap = 22 if not compact else 12
    # Vertical budget: title (y=18) + legend row (y=42) -> first tensor row below.
    top = 74 if not compact else 52
    left = 250
    barw = width - left - 130

    # Four tensors: (name, dims-label, has_layer_dim, touched, note)
    rows = [
        (
            "req_to_device_buffer",
            f"[R, {padded}]",
            False,
            False,
            "physical slot ids &middot; no layer axis",
        ),
        (
            "req_device_buffer_tokens",
            f"[L={layer_num}, R, {padded}]",
            True,
            True,
            "reserved col &rarr; B<tspan>l</tspan>",
        ),
        (
            "req_device_buffer_token_locs",
            f"[L={layer_num}, R, {padded}]",
            True,
            True,
            "reserved col &rarr; B<tspan>l</tspan>",
        ),
        (
            "lru_slots",
            f"[L={layer_num}, R, {padded}]",
            True,
            False,
            "arange&rarr;{padded}; kernel scans [0,B<tspan>l</tspan>)",
        ),
        (
            "top_k_device_locs",
            f"[R, {top_k}]",
            False,
            False,
            "top_k-sized &middot; unaffected",
        ),
    ]

    h = top + len(rows) * (row_h + gap) + (46 if not compact else 22)
    svg = [f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px">']

    # Title (row 1) and legend (row 2) on separate lines so they never overlap.
    if not compact:
        svg.append(
            _txt(
                0,
                18,
                "Tensor shapes on the touched HiSparse kernel path "
                "(R = max concurrent requests, L = indexer layers)",
                fill="#9aa4b2",
                size=12,
                weight="600",
            )
        )
    ly = 44 if not compact else 30  # legend baseline, its own row
    sub = '<tspan baseline-shift="sub" font-size="8">l</tspan>'
    # Chip widths sized to their labels to avoid crowding.
    chips = [
        (NV, f"modified (per-layer B{sub})", 190),
        (GREY, "unchanged shape", 150),
        (AMBER, f"reserved slot &rarr; col B{sub}", 0),
    ]
    cx = 0
    for fill, label, advance in chips:
        svg.append(_bar(cx, ly - 10, 12, 12, fill, rx=2))
        svg.append(_txt(cx + 18, ly, label, fill="#c9d2dd", size=10.5))
        cx += advance

    y = top
    for name, dims, has_layer, touched, note in rows:
        fill = "#1d2937" if touched else PANEL
        # name + dims
        svg.append(
            _txt(
                0,
                y + row_h * 0.62,
                name,
                fill="#e7ebf0",
                size=12.5,
                weight="700",
                mono=True,
            )
        )
        svg.append(
            _txt(
                0,
                y + row_h * 0.62 + 15,
                dims,
                fill=(NV if touched else "#9aa4b2"),
                size=11,
                mono=True,
            )
        )

        # the bar (represents the padded width). Draw the "logical B_l" region + padding + reserved slot.
        # scale: full bar = padded.
        def sx(v):
            return left + v / padded * barw

        # padding region (light)
        svg.append(
            _bar(left, y, barw, row_h, fill, stroke=(NV if touched else "#2c333d"))
        )
        if touched:
            # uniform B region (up to b_uniform) and the extra DP headroom to max_b
            wU = sx(b_uniform) - left
            svg.append(_bar(left, y, wU, row_h, "#25350a", stroke=NV))
            svg.append(
                _txt(
                    left + 6,
                    y + row_h * 0.4,
                    "logical B",
                    fill=NV,
                    size=10,
                    weight="700",
                )
            )
            svg.append(
                _txt(
                    left + 6,
                    y + row_h * 0.4 + 13,
                    f"uniform={b_uniform} &middot; dp&isin;[{min(b_dp)},{max(b_dp)}]",
                    fill="#a9c46c",
                    size=9.5,
                    mono=True,
                )
            )
            # reserved slot marker at column B_l (draw a thin amber slab near b range)
            rxk = sx(max_b)
            svg.append(
                f'<rect x="{rxk:.1f}" y="{y}" width="6" height="{row_h}" fill="{AMBER}"/>'
            )
            svg.append(
                _txt(rxk + 10, y + row_h * 0.62, "reserved @ B", fill=AMBER, size=9.5)
            )
        else:
            svg.append(
                _txt(
                    left + 8,
                    y + row_h * 0.62,
                    ("has layer dim &middot; " if has_layer else "")
                    + "shape unchanged",
                    fill="#8892a0",
                    size=10,
                )
            )
        # right-side note
        svg.append(
            _txt(
                left + barw + 10,
                y + row_h * 0.62,
                note.replace("{padded}", str(padded)),
                fill="#9aa4b2",
                size=9.5,
            )
        )
        y += row_h + gap

    # bottom caption
    if not compact:
        svg.append(
            _txt(
                0,
                h - 8,
                f"Only the kernel template hot_buffer_size and the reserved-slot column change per layer; "
                f"all physical widths stay at MAX(B_l)+page = {max_b}+{page_size} = {padded}. No tensor is reshaped.",
                fill="#9aa4b2",
                size=10.5,
            )
        )
    svg.append("</svg>")
    return "".join(svg)
