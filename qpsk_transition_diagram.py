#!/usr/bin/env python3
"""Create three separate QPSK phase-transition diagrams.

Each output shows a transition from the 45-degree state to one of the other
three QPSK states, with a carrier waveform and constellation snapshots.

Run: python3 qpsk_transition_diagram.py
Optional: python3 qpsk_transition_diagram.py --output-dir figures --cycles 10
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# Use a writable font cache and a non-GUI renderer for reliable IDE execution.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-qpsk-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from modulation_spectrum import plot_spectra, qpsk_envelope

# Gray-coded labels for this illustrative four-state QPSK constellation.
# In the 2.4 GHz IEEE 802.15.4 PHY, four input bits select a 32-chip DSSS
# sequence that is then O-QPSK modulated; these pairs are not PHY symbols.
STATE_BITS = {45: "01", 135: "00", 225: "10", 315: "11"}
BITSTREAM = "0110100111001011"


def make_transition_figure(target_deg, label, cycles, carrier_ghz):
    start_deg = 45.0
    snapshots = np.linspace(0.0, 1.0, 5)
    n_samples = max(5000, int(cycles * 1200))
    t = np.linspace(-2.0, cycles + 2.0, n_samples)

    # Move in a straight line through the constellation's Cartesian plane. For the antipodal state,
    # the direct line between the points passes through the origin.
    progress = np.clip(t / cycles, 0.0, 1.0)
    smooth_progress = 0.5 - 0.5 * np.cos(np.pi * progress)
    start_angle = np.deg2rad(start_deg)
    target_angle = np.deg2rad(target_deg)
    start_x, start_y = np.cos(start_angle), np.sin(start_angle)
    target_x, target_y = np.cos(target_angle), np.sin(target_angle)
    x_component = start_x + (target_x - start_x) * smooth_progress
    y_component = start_y + (target_y - start_y) * smooth_progress
    # s(t) = X(t) cos(wt) - Y(t) sin(wt). Since the vector follows a straight
    # chord between constellation points, the signal envelope is not constant.
    carrier_phase = 2.0 * np.pi * t
    waveform = x_component * np.cos(carrier_phase) - y_component * np.sin(carrier_phase)

    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.7])
    ax = fig.add_subplot(grid[0])
    ns_per_cycle = 1.0 / carrier_ghz
    ax.plot(t * ns_per_cycle, waveform, lw=2.2, color="#1f77b4")
    ax.axvspan(0, cycles * ns_per_cycle, color="#ffb366", alpha=0.2,
               label=f"{cycles:g}-cycle constellation transition")
    ax.axhline(0, color="0.55", lw=0.8)

    for idx, fraction in enumerate(snapshots):
        cycle_x = cycles * fraction
        x = cycle_x * ns_per_cycle
        amount = 0.5 - 0.5 * np.cos(np.pi * fraction)
        snap_x = start_x + (target_x - start_x) * amount
        snap_y = start_y + (target_y - start_y) * amount
        y = snap_x * np.cos(2.0 * np.pi * cycle_x) - snap_y * np.sin(2.0 * np.pi * cycle_x)
        ax.axvline(x, color="0.45", lw=0.9, ls=":")
        ax.scatter([x], [y], color="#d62728", s=52, zorder=4,
                   edgecolor="white", linewidth=0.5)
        ax.text(x, 1.19, str(idx + 1), ha="center", va="center",
                color="#d62728", fontsize=10)

    ax.set_xlim(-2 * ns_per_cycle, (cycles + 2) * ns_per_cycle)
    ax.set_ylim(-1.35, 1.4)
    ax.set_ylabel("Signal amplitude")
    ax.set_xlabel(f"Time (ns; {carrier_ghz:g} GHz carrier)")
    tick_step = 1 if cycles <= 12 else cycles / 10
    cycle_ticks = np.arange(0, cycles + 0.01, tick_step)
    ax.set_xticks(cycle_ticks * ns_per_cycle,
                  [f"{tick * ns_per_cycle:g}" for tick in cycle_ticks])
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", framealpha=0.9)

    snapshots_grid = grid[1].subgridspec(1, len(snapshots), wspace=0.32)
    qpsk_angles = np.deg2rad([45, 135, 225, 315])
    qpsk_xy = np.column_stack((np.cos(qpsk_angles), np.sin(qpsk_angles)))
    for snap_idx, fraction in enumerate(snapshots):
        cax = fig.add_subplot(snapshots_grid[0, snap_idx])
        amount = 0.5 - 0.5 * np.cos(np.pi * fraction)
        snap_x = start_x + (target_x - start_x) * amount
        snap_y = start_y + (target_y - start_y) * amount
        cax.scatter(qpsk_xy[:, 0], qpsk_xy[:, 1], s=42, color="#c8c8c8", zorder=2)
        # Show the square-edge chord (or the direct diagonal through the origin).
        cax.plot([start_x, target_x], [start_y, target_y], color="#d62728", lw=1.5, alpha=0.45)
        cax.scatter([snap_x], [snap_y], s=115, color="#d62728",
                    edgecolor="black", linewidth=0.9, zorder=3)
        cax.axhline(0, color="0.78", lw=0.8)
        cax.axvline(0, color="0.78", lw=0.8)
        cax.set_xlim(-1.25, 1.25)
        cax.set_ylim(-1.25, 1.25)
        cax.set_aspect("equal", adjustable="box")
        for degree, (state_x, state_y) in zip((45, 135, 225, 315), qpsk_xy):
            cax.annotate(STATE_BITS[degree], (state_x, state_y),
                         xytext=(0, 8 if state_y > 0 else -13),
                         textcoords="offset points", ha="center",
                         va="bottom" if state_y > 0 else "top", fontsize=9)
        cax.set_xticks([-1, 0, 1])
        cax.set_yticks([-1, 0, 1])
        cax.tick_params(labelsize=9)
        cax.grid(True, alpha=0.18)
        cax.set_title(f"{fraction:.0%}\n{fraction * cycles:g} cycles", fontsize=11)

    fig.suptitle(
        f"QPSK phase transition: 45° → {target_deg:.0f}° ({label})\n"
        f"Raised-cosine Cartesian-coordinate interpolation over {cycles:g} carrier cycles ({carrier_ghz:g} GHz carrier)",
        fontsize=16,
    )
    return fig


def make_transition_overview():
    """Draw all six bidirectional state pairs: four edges and two diagonals."""
    angles = np.deg2rad([45, 135, 225, 315])
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # Connect each distinct pair once with a two-headed arrow. Adjacent states
    # form the square; opposite states form the X through the origin.
    for i in range(4):
        j = (i + 1) % 4
        ax.annotate("", xy=points[j], xytext=points[i],
                    arrowprops=dict(arrowstyle="<->", color="#2878b5", lw=2.1,
                                    shrinkA=18, shrinkB=18, mutation_scale=14))
    for i, j in ((0, 2), (1, 3)):
        ax.annotate("", xy=points[j], xytext=points[i],
                    arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2.1,
                                    shrinkA=18, shrinkB=18, mutation_scale=14))

    ax.scatter(points[:, 0], points[:, 1], s=190, color="#d62728",
               edgecolor="black", linewidth=1.2, zorder=4)
    label_offsets = [(12, 10), (-12, 10), (-12, -22), (12, -22)]
    for (x, y), degree, offset in zip(points, (45, 135, 225, 315), label_offsets):
        ax.annotate(f"{STATE_BITS[degree]}\n{degree}°", (x, y), xytext=offset, textcoords="offset points",
                    ha="left" if offset[0] > 0 else "right",
                    va="bottom" if offset[1] > 0 else "top",
                    fontsize=12, weight="bold", zorder=5)

    # Label the origin to emphasize that the diagonal transitions pass through it.
    ax.scatter([0], [0], s=28, color="black", zorder=3)
    ax.annotate("origin", (0, 0), xytext=(0.1, 0.0), textcoords="data",
                ha="left", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.92))
    ax.plot([], [], color="#2878b5", lw=2.1, label="Adjacent-state transitions (square edges)")
    ax.plot([], [], color="#d62728", lw=2.1, label="Opposite-state transitions (through origin)")
    ax.set_title("QPSK constellation: all possible state transitions\n"
                 "Illustrative bit labels; 2.4 GHz IEEE 802.15.4 uses 4-bit to 32-chip DSSS before O-QPSK",
                 fontsize=13, pad=16)
    ax.set_xlabel("Constellation x-coordinate")
    ax.set_ylabel("Constellation y-coordinate")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), framealpha=0.95)
    return fig


def make_bitstream_figure(carrier_ghz):
    """Show a 16-bit stream as eight Gray-coded QPSK symbols."""
    bitstream = BITSTREAM
    symbols = [bitstream[i:i + 2] for i in range(0, len(bitstream), 2)]
    degree_for_bits = {bits: degree for degree, bits in STATE_BITS.items()}
    symbol_degrees = [degree_for_bits[bits] for bits in symbols]
    symbol_angles = np.deg2rad(symbol_degrees)
    symbol_points = np.column_stack((np.cos(symbol_angles), np.sin(symbol_angles)))

    cycles_per_symbol = 6
    duration = len(symbols) * cycles_per_symbol
    t = np.linspace(0, duration, max(8000, duration * 500))
    x_component = np.empty_like(t)
    y_component = np.empty_like(t)
    for symbol_idx, (bits, target) in enumerate(zip(symbols, symbol_points)):
        start = symbol_idx * cycles_per_symbol
        end = start + cycles_per_symbol
        mask = (t >= start) & (t < end)
        if symbol_idx == 0:
            source = target
        else:
            source = symbol_points[symbol_idx - 1]
        amount = (t[mask] - start) / cycles_per_symbol
        amount = 0.5 - 0.5 * np.cos(np.pi * amount)
        x_component[mask] = source[0] + (target[0] - source[0]) * amount
        y_component[mask] = source[1] + (target[1] - source[1]) * amount
    x_component[-1], y_component[-1] = symbol_points[-1]

    waveform = (x_component * np.cos(2 * np.pi * t)
                - y_component * np.sin(2 * np.pi * t))
    time_ns = t / carrier_ghz
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.35, 1.0, 1.35, 1.0])
    for row, first_symbol in enumerate((0, 4)):
        ax = fig.add_subplot(grid[row * 2])
        left = first_symbol * cycles_per_symbol
        right = left + 4 * cycles_per_symbol
        segment = (t >= left) & (t <= right)
        left_ns = left / carrier_ghz
        right_ns = right / carrier_ghz
        ax.plot(time_ns[segment] - left_ns, waveform[segment], color="#1f77b4", lw=1.5)
        ax.axhline(0, color="0.55", lw=0.8)
        for idx in range(first_symbol, first_symbol + 4):
            start = (idx - first_symbol) * cycles_per_symbol / carrier_ghz
            end = start + cycles_per_symbol / carrier_ghz
            ax.axvline(start, color="0.55", lw=0.8, ls=":")
            ax.axvspan(start, end, color="#ffb366", alpha=0.11 if idx % 2 == 0 else 0.2)
            ax.text((start + end) / 2, 1.18, f"{idx + 1}: {symbols[idx]}",
                    ha="center", va="center", fontsize=10, weight="bold")
        ax.axvline(4 * cycles_per_symbol / carrier_ghz, color="0.55", lw=0.8, ls=":")
        ax.set_xlim(0, 4 * cycles_per_symbol / carrier_ghz)
        ax.set_ylim(-1.35, 1.4)
        ax.set_ylabel("Signal amplitude")
        ax.set_xlabel(f"Time (ns; {carrier_ghz:g} GHz carrier)")
        ax.set_title(f"Symbols {first_symbol + 1}–{first_symbol + 4}")
        ax.grid(True, alpha=0.22)

    angles = np.deg2rad([45, 135, 225, 315])
    constellation = np.column_stack((np.cos(angles), np.sin(angles)))
    snapshots_grids = {
        1: grid[1].subgridspec(1, 4, wspace=0.24),
        3: grid[3].subgridspec(1, 4, wspace=0.24),
    }
    for idx, (bits, point) in enumerate(zip(symbols, symbol_points)):
        snapshot_row = 1 if idx < 4 else 3
        snapshot_col = idx % 4
        cax = fig.add_subplot(snapshots_grids[snapshot_row][0, snapshot_col])
        cax.scatter(constellation[:, 0], constellation[:, 1], s=45,
                    color="#c8c8c8", zorder=2)
        if idx > 0:
            previous = symbol_points[idx - 1]
            cax.plot([previous[0], point[0]], [previous[1], point[1]],
                     color="#d62728", lw=1.7, alpha=0.55)
        cax.scatter([point[0]], [point[1]], s=135, color="#d62728",
                    edgecolor="black", linewidth=0.9, zorder=3)
        for degree, (state_x, state_y) in zip((45, 135, 225, 315), constellation):
            cax.annotate(STATE_BITS[degree], (state_x, state_y),
                         xytext=(0, 8 if state_y > 0 else -13),
                         textcoords="offset points", ha="center",
                         va="bottom" if state_y > 0 else "top", fontsize=9)
        cax.axhline(0, color="0.78", lw=0.8)
        cax.axvline(0, color="0.78", lw=0.8)
        cax.set_xlim(-1.3, 1.3)
        cax.set_ylim(-1.3, 1.3)
        cax.set_aspect("equal", adjustable="box")
        cax.set_xticks([-1, 0, 1])
        cax.set_yticks([-1, 0, 1])
        cax.tick_params(labelsize=8)
        cax.grid(True, alpha=0.18)
        cax.set_title(f"Step {idx + 1}: {bits}", fontsize=11)

    fig.suptitle(
        f"16-bit QPSK stream: {' '.join(bitstream[i:i + 4] for i in range(0, 16, 4))} "
        f"({carrier_ghz:g} GHz carrier)\n"
        "Gray-coded 2-bit symbols; smoothed Cartesian transitions for illustration",
        fontsize=15,
    )
    return fig


def make_bitstream_spectrum_figure(carrier_ghz):
    # Use a long seeded stream only for PSD estimation. The visible waveform
    # remains the original 16-bit example.
    rng = np.random.default_rng(20261019)
    states = list(STATE_BITS.values())
    symbols = [states[idx] for idx in rng.integers(0, len(states), size=2048)]
    samples_per_cycle = 32
    _, envelope = qpsk_envelope(
        symbols, STATE_BITS, cycles_per_symbol=6,
        samples_per_cycle=samples_per_cycle,
    )
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    plot_spectra(ax, [
        ("QPSK spectrum", envelope, "#2878b5", "-"),
    ], samples_per_cycle=samples_per_cycle, carrier_ghz=carrier_ghz)
    ax.set_title(f"QPSK bitstream spectrum around {carrier_ghz:g} GHz")
    ax.text(0.01, 0.97, "Welch-averaged 4096-bit stream (not shown); normalized to in-band power",
            transform=ax.transAxes, fontsize=8, color="0.35", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))
    return fig


def open_image(image_path):
    image_path = str(image_path.resolve())
    if sys.platform == "darwin":
        subprocess.Popen(["open", image_path])
    elif os.name == "nt":
        os.startfile(image_path)
    else:
        subprocess.Popen(["xdg-open", image_path])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".",
                        help="Folder for the three PNG files (default: current folder)")
    parser.add_argument("--cycles", type=float, default=10.0,
                        help="Transition duration in carrier cycles (default: %(default)s)")
    parser.add_argument("--carrier-ghz", type=float, default=2.45,
                        help="Carrier frequency used for waveforms and spectra")
    parser.add_argument("--no-open", action="store_true",
                        help="Save the PNGs without opening them")
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [(135.0, "+90°", 135), (225.0, "+180°", 225), (315.0, "−90°", 315)]
    output_paths = []
    for target_deg, label, suffix in destinations:
        fig = make_transition_figure(target_deg, label,
                                     args.cycles, args.carrier_ghz)
        path = output_dir / f"qpsk_transition_45_to_{suffix}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)
        print(f"Saved {path.resolve()}")

    overview = make_transition_overview()
    overview_path = output_dir / "qpsk_constellation_transitions.png"
    overview.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(overview)
    output_paths.append(overview_path)
    print(f"Saved {overview_path.resolve()}")

    bitstream_figure = make_bitstream_figure(args.carrier_ghz)
    bitstream_path = output_dir / "qpsk_bitstream_16bits.png"
    bitstream_figure.savefig(bitstream_path, dpi=180, bbox_inches="tight")
    plt.close(bitstream_figure)
    output_paths.append(bitstream_path)
    print(f"Saved {bitstream_path.resolve()}")

    spectrum = make_bitstream_spectrum_figure(args.carrier_ghz)
    spectrum_path = output_dir / "qpsk_bitstream_spectrum.png"
    spectrum.savefig(spectrum_path, dpi=180, bbox_inches="tight")
    plt.close(spectrum)
    output_paths.append(spectrum_path)
    print(f"Saved {spectrum_path.resolve()}")

    if not args.no_open:
        for path in output_paths:
            open_image(path)


if __name__ == "__main__":
    main()
