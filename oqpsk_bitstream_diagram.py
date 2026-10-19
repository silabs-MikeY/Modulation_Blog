#!/usr/bin/env python3
"""Illustrate OQPSK modulation of the same 16-bit example stream.

Run: python3 oqpsk_bitstream_diagram.py
Optional: python3 oqpsk_bitstream_diagram.py --output-dir figures --no-open
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-oqpsk-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from modulation_spectrum import (
    oqpsk_components, pa_soft_compression, plot_spectra, qpsk_envelope,
)

# Same illustrative Gray mapping as qpsk_transition_diagram.py.
STATE_BITS = {45: "01", 135: "00", 225: "10", 315: "11"}
BITSTREAM = "0110100111001011"


def make_oqpsk_figure(carrier_ghz):
    ns_per_cycle = 1.0 / carrier_ghz
    symbols = [BITSTREAM[i:i + 2] for i in range(0, len(BITSTREAM), 2)]
    degree_for_bits = {bits: degree for degree, bits in STATE_BITS.items()}
    degrees = [degree_for_bits[bits] for bits in symbols]
    angles = np.deg2rad(degrees)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    i_levels, q_levels = points[:, 0], points[:, 1]

    cycles_per_symbol = 6
    symbol_duration = len(symbols)
    duration = symbol_duration * cycles_per_symbol
    t, i_signal, q_signal = oqpsk_components(
        symbols, STATE_BITS, cycles_per_symbol=cycles_per_symbol)
    carrier = i_signal * np.cos(2 * np.pi * t) - q_signal * np.sin(2 * np.pi * t)
    time_ns = t * ns_per_cycle

    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.1, 2.0, 1.1])
    state_angles = np.deg2rad([45, 135, 225, 315])
    constellation = np.column_stack((np.cos(state_angles), np.sin(state_angles)))

    # Two time-domain rows, each followed by the constellation steps for
    # every symbol change shown in that part of the waveform.
    for block, first_symbol in enumerate((0, 4)):
        time_grid = grid[block * 2].subgridspec(2, 1, height_ratios=[1.55, 0.58], hspace=0.04)
        left = first_symbol * cycles_per_symbol
        right = left + 4 * cycles_per_symbol
        mask = (t >= left) & (t <= right)

        ax = fig.add_subplot(time_grid[0])
        left_ns = left * ns_per_cycle
        ax.plot(time_ns[mask] - left_ns, carrier[mask], color="#1f77b4", lw=1.35)
        ax.axhline(0, color="0.55", lw=0.8)
        for idx in range(first_symbol, first_symbol + 4):
            cell_left = (idx - first_symbol) * cycles_per_symbol * ns_per_cycle
            cell_right = cell_left + cycles_per_symbol * ns_per_cycle
            ax.axvspan(cell_left, cell_right, color="#ffb366",
                       alpha=0.13 if idx % 2 == 0 else 0.22)
            ax.axvline(cell_left, color="0.55", lw=0.8, ls=":")
            ax.text((cell_left + cell_right) / 2, 1.18,
                    f"{idx + 1}: {symbols[idx]}", ha="center", va="center",
                    fontsize=10, weight="bold")
        ax.axvline(4 * cycles_per_symbol * ns_per_cycle, color="0.55", lw=0.8, ls=":")
        ax.set_xlim(0, 4 * cycles_per_symbol * ns_per_cycle)
        ax.set_ylim(-1.35, 1.4)
        ax.set_ylabel("RF amplitude")
        ax.set_title(f"OQPSK carrier, symbols {first_symbol + 1}–{first_symbol + 4}")
        ax.grid(True, alpha=0.22)

        b_ax = fig.add_subplot(time_grid[1], sharex=ax)
        b_ax.plot(time_ns[mask] - left_ns, i_signal[mask], color="#2878b5",
                  lw=1.8, label="bit1")
        b_ax.plot(time_ns[mask] - left_ns, q_signal[mask], color="#e07a24",
                  lw=1.8, label="bit0 (½-symbol delay)")
        for idx in range(1, 4):
            b_ax.axvline(idx * cycles_per_symbol * ns_per_cycle, color="#2878b5", alpha=0.22,
                         lw=0.8, ls=":")
            b_ax.axvline((idx + 0.5) * cycles_per_symbol * ns_per_cycle, color="#e07a24",
                         alpha=0.25, lw=0.8, ls=":")
        b_ax.set_xlim(0, 4 * cycles_per_symbol * ns_per_cycle)
        b_ax.set_ylim(-1.3, 1.3)
        b_ax.set_yticks([-1, 1], labels=["−1", "+1"])
        b_ax.set_ylabel("Bit level")
        b_ax.set_xlabel(f"Time (ns; {carrier_ghz:g} GHz carrier)")
        b_ax.legend(loc="upper right", ncol=2, framealpha=0.95)
        b_ax.grid(True, axis="y", alpha=0.22)

        # First row contains transitions 1→2 through 3→4. The second row
        # begins with 4→5 at its left edge and continues through 7→8.
        transition_grid = grid[block * 2 + 1].subgridspec(1, 4, wspace=0.22)
        for panel_idx in range(4):
            idx = first_symbol + panel_idx
            cax = fig.add_subplot(transition_grid[0, panel_idx])
            cax.scatter(constellation[:, 0], constellation[:, 1], s=42,
                        color="#c8c8c8", zorder=2)
            for degree, (sx, sy) in zip((45, 135, 225, 315), constellation):
                cax.annotate(STATE_BITS[degree], (sx, sy),
                             xytext=(0, 5 if sy > 0 else -9), textcoords="offset points",
                             ha="center", va="bottom" if sy > 0 else "top", fontsize=9)

            if idx == 0:
                source = target = points[0]
                cax.scatter([source[0]], [source[1]], s=88, color="#d62728",
                            edgecolor="black", linewidth=0.7, zorder=4)
                panel_title = f"Start: {symbols[0]}"
            else:
                source = points[idx - 1]
                intermediate = np.array([points[idx, 0], source[1]])
                target = points[idx]
                if np.allclose(source, target):
                    cax.scatter([source[0]], [source[1]], s=88, color="#d62728",
                                edgecolor="black", linewidth=0.7, zorder=4)
                    panel_title = f"Hold: {symbols[idx]}"
                else:
                    # Sample the gradual edge-by-edge move at the start,
                    # quarter points, midpoint, three-quarter point, and end.
                    progress_samples = np.linspace(0, 1, 5)
                    path_samples = []
                    for progress in progress_samples:
                        if progress <= 0.5:
                            local = 0.5 - 0.5 * np.cos(np.pi * progress * 2)
                            point = source + (intermediate - source) * local
                        else:
                            local_progress = (progress - 0.5) * 2
                            local = 0.5 - 0.5 * np.cos(np.pi * local_progress)
                            point = intermediate + (target - intermediate) * local
                        path_samples.append(point)

                    for edge_start, edge_end, color in (
                            (source, intermediate, "#2878b5"),
                            (intermediate, target, "#e07a24")):
                        if not np.allclose(edge_start, edge_end):
                            cax.annotate("", xy=edge_end, xytext=edge_start,
                                         arrowprops=dict(arrowstyle="-|>", color=color,
                                                         lw=1.7, shrinkA=5, shrinkB=5,
                                                         mutation_scale=10))
                    for sample_idx, (sample, color) in enumerate(zip(
                            path_samples, ("#333333", "#6497c4", "#e09a58", "#d95f59", "#d62728"))):
                        cax.scatter([sample[0]], [sample[1]], s=54 if sample_idx not in (0, 4) else 82,
                                    color=color, edgecolor="white" if sample_idx == 0 else "none",
                                    linewidth=0.6, zorder=4 + sample_idx * 0.01)
                    panel_title = f"{idx}→{idx + 1}: {symbols[idx - 1]} → {symbols[idx]}"
            cax.axhline(0, color="0.78", lw=0.7)
            cax.axvline(0, color="0.78", lw=0.7)
            cax.set_xlim(-1.42, 1.42)
            cax.set_ylim(-1.42, 1.42)
            cax.set_aspect("equal", adjustable="box")
            cax.set_xticks([])
            cax.set_yticks([])
            cax.grid(True, alpha=0.16)
            cax.set_title(panel_title, fontsize=10)

    fig.suptitle(
        f"OQPSK, 16-bit stream: {' '.join(BITSTREAM[i:i + 4] for i in range(0, 16, 4))} "
        f"({carrier_ghz:g} GHz carrier)\n"
        "Gradual bit1 change followed by bit0, offset by half a symbol",
        fontsize=15,
    )
    return fig


def make_bitstream_spectrum_figure(carrier_ghz):
    # Keep the longer seeded data private to the spectrum estimate. The
    # displayed time-domain example remains the same 16-bit stream.
    rng = np.random.default_rng(20261019)
    states = list(STATE_BITS.values())
    symbols = [states[idx] for idx in rng.integers(0, len(states), size=2048)]
    samples_per_cycle = 32
    _, bit1, bit0 = oqpsk_components(
        symbols, STATE_BITS, cycles_per_symbol=6,
        samples_per_cycle=samples_per_cycle,
    )
    oqpsk = bit1 + 1j * bit0
    _, qpsk = qpsk_envelope(
        symbols, STATE_BITS, cycles_per_symbol=6,
        samples_per_cycle=samples_per_cycle,
    )

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    plot_spectra(ax, [
        ("QPSK", qpsk, "#2878b5", "-"),
        ("OQPSK", oqpsk, "#e07a24", "-"),
    ], samples_per_cycle=samples_per_cycle, carrier_ghz=carrier_ghz)
    ax.set_title(f"OQPSK and QPSK spectra around {carrier_ghz:g} GHz")
    ax.text(0.01, 0.97, "Welch-averaged 4096-bit stream (not shown); spectra normalized to in-band power",
            transform=ax.transAxes, fontsize=8, color="0.35", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))
    return fig


def make_pa_spectrum_figure(carrier_ghz):
    """Compare linear and compressed QPSK/OQPSK spectra for the hidden long stream."""
    rng = np.random.default_rng(20261019)
    states = list(STATE_BITS.values())
    symbols = [states[idx] for idx in rng.integers(0, len(states), size=2048)]
    samples_per_cycle = 32
    _, bit1, bit0 = oqpsk_components(
        symbols, STATE_BITS, cycles_per_symbol=6,
        samples_per_cycle=samples_per_cycle,
    )
    oqpsk = bit1 + 1j * bit0
    _, qpsk = qpsk_envelope(
        symbols, STATE_BITS, cycles_per_symbol=6,
        samples_per_cycle=samples_per_cycle,
    )

    # Match the gain, saturation output, and smoothness of the illustrative
    # AM/AM curve in rf_amplification_gain.py.
    gain, saturation_output, smoothness = 3.0, 2.4, 2.0
    qpsk_pa = pa_soft_compression(qpsk, gain, saturation_output, smoothness)
    oqpsk_pa = pa_soft_compression(oqpsk, gain, saturation_output, smoothness)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    plot_spectra(ax, [
        ("QPSK, after PA", qpsk_pa, "#c62828", "-"),
        ("OQPSK, after PA", oqpsk_pa, "#2e8b57", "-"),
    ], samples_per_cycle=samples_per_cycle, carrier_ghz=carrier_ghz)
    ax.set_title(f"QPSK and OQPSK spectra after PA compression ({carrier_ghz:g} GHz)")
    ax.text(
        0.01, 0.97,
        "Hidden 4096-bit stream; AM/AM: G=3, saturation output=2.4, smoothness=2",
        transform=ax.transAxes, fontsize=8, color="0.35", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )
    return fig


def make_transition_overview():
    """Show every allowed single-branch OQPSK transition between states."""
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    degrees = (45, 135, 225, 315)
    angles = np.deg2rad(degrees)
    points = np.column_stack((np.cos(angles), np.sin(angles)))

    # Horizontal edges toggle bit1; vertical edges toggle bit0. Each is
    # bidirectional. A diagonal symbol change is reached by two such steps.
    for i, j in ((0, 1), (2, 3)):
        ax.annotate("", xy=points[j], xytext=points[i],
                    arrowprops=dict(arrowstyle="<->", color="#2878b5", lw=2.4,
                                    shrinkA=16, shrinkB=16, mutation_scale=15))
    for i, j in ((0, 3), (1, 2)):
        ax.annotate("", xy=points[j], xytext=points[i],
                    arrowprops=dict(arrowstyle="<->", color="#e07a24", lw=2.4,
                                    shrinkA=16, shrinkB=16, mutation_scale=15))

    ax.scatter(points[:, 0], points[:, 1], s=185, color="#d62728",
               edgecolor="black", linewidth=1.1, zorder=4)
    offsets = [(12, 10), (-12, 10), (-12, -22), (12, -22)]
    for (x, y), degree, offset in zip(points, degrees, offsets):
        ax.annotate(f"{STATE_BITS[degree]}\n{degree}°", (x, y),
                    xytext=offset, textcoords="offset points",
                    ha="left" if offset[0] > 0 else "right",
                    va="bottom" if offset[1] > 0 else "top",
                    fontsize=12, weight="bold")

    ax.plot([], [], color="#2878b5", lw=2.4, label="bit1 transition (horizontal branch)")
    ax.plot([], [], color="#e07a24", lw=2.4, label="bit0 transition (vertical branch, delayed)")
    ax.set_title("OQPSK constellation transitions\n"
                 "Any symbol change occurs as one or two single-branch steps",
                 fontsize=14, pad=16)
    ax.set_xlabel("bit1 branch level")
    ax.set_ylabel("bit0 branch level")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), framealpha=0.95)
    return fig


def open_image(path):
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path.resolve())])
    elif os.name == "nt":
        os.startfile(str(path.resolve()))
    else:
        subprocess.Popen(["xdg-open", str(path.resolve())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".", help="Folder for the PNG (default: current folder)")
    parser.add_argument("--carrier-ghz", type=float, default=2.45,
                        help="Carrier frequency used for waveforms and spectra")
    parser.add_argument("--no-open", action="store_true", help="Save without opening the PNG")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "oqpsk_bitstream_16bits.png"
    fig = make_oqpsk_figure(args.carrier_ghz)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.resolve()}")
    overview_path = output_dir / "oqpsk_constellation_transitions.png"
    overview = make_transition_overview()
    overview.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(overview)
    print(f"Saved {overview_path.resolve()}")
    spectrum_path = output_dir / "oqpsk_bitstream_spectrum.png"
    spectrum = make_bitstream_spectrum_figure(args.carrier_ghz)
    spectrum.savefig(spectrum_path, dpi=180, bbox_inches="tight")
    plt.close(spectrum)
    print(f"Saved {spectrum_path.resolve()}")
    pa_spectrum_path = output_dir / "oqpsk_bitstream_spectrum_pa.png"
    pa_spectrum = make_pa_spectrum_figure(args.carrier_ghz)
    pa_spectrum.savefig(pa_spectrum_path, dpi=180, bbox_inches="tight")
    plt.close(pa_spectrum)
    print(f"Saved {pa_spectrum_path.resolve()}")
    if not args.no_open:
        open_image(path)
        open_image(overview_path)
        open_image(spectrum_path)
        open_image(pa_spectrum_path)


if __name__ == "__main__":
    main()
