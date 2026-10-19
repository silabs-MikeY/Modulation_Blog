#!/usr/bin/env python3
"""Visualize half-sine chip shaping for an illustrative 802.15.4 codeword.

Run:
    python3 dsss_half_sine_chip_shaping.py
    python3 dsss_half_sine_chip_shaping.py --no-open
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-dsss-chip-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHIPS_0000 = "11011001110000110101001000101110"  # c0 first, all 32 chips.
CHIP_RATE = 2e6


def half_sine_pulse(tau):
    """IEEE 802.15.4 half-sine chip pulse, with time measured in chip periods."""
    return np.where((tau >= 0.0) & (tau <= 2.0), np.sin(np.pi * tau / 2.0), 0.0)


def make_figure():
    tc_us = 1e6 / CHIP_RATE
    chips = np.array([1.0 if bit == "1" else -1.0 for bit in CHIPS_0000])
    chip_count = len(chips)
    samples_per_chip = 250
    t_chip = np.linspace(0.0, chip_count, chip_count * samples_per_chip + 1)
    i_signal = np.zeros_like(t_chip)
    q_signal = np.zeros_like(t_chip)

    # Chips enter alternately in I and Q. Each pulse lasts 2Tc; Q begins one
    # chip period (Tc) after I, so neighboring I/Q half-sines overlap.
    for k, chip in enumerate(chips):
        pulse = chip * half_sine_pulse(t_chip - k)
        if k % 2 == 0:
            i_signal += pulse
        else:
            q_signal += pulse
    complex_envelope = i_signal + 1j * q_signal
    envelope = np.abs(complex_envelope)
    phase_deg = np.rad2deg(np.unwrap(np.angle(complex_envelope)))

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), constrained_layout=True,
                              gridspec_kw={"height_ratios": [0.8, 1.5, 0.9, 0.65]})

    # 1) Isolated pulse shape. Do not overlay a rectangular pulse here: it was
    #    mistaken for the transmitted signal in the earlier explanatory plot.
    ax = axes[0]
    t_single = np.linspace(0, 2, 700)
    pulse = half_sine_pulse(t_single)
    ax.plot(t_single, pulse, color="#2878b5", lw=2.5,
            label=r"Half-sine chip pulse: $p(t)=\sin(\pi t/(2T_c))$")
    ax.axvspan(0, 1, color="#f1c75b", alpha=0.18, label="One chip period, $T_c$")
    ax.axvline(2, color="0.45", lw=0.8, ls=":")
    ax.set_xlim(0, 2.15)
    ax.set_ylim(-0.06, 1.18)
    ax.set_ylabel("Pulse amplitude")
    ax.set_xlabel(r"Time ($T_c$ units; pulse spans $2T_c$)")
    ax.set_title("One chip contribution: a smooth pulse lasting two chip periods")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", ncol=1, fontsize=9, framealpha=0.95)

    # 2) Actual overlapping I/Q pulse contributions for the full 32-chip codeword.
    ax = axes[1]
    ax.plot(t_chip, i_signal, color="#2878b5", lw=2.0, label="I component")
    ax.plot(t_chip, q_signal, color="#e07a24", lw=2.0, label="Q component (offset by $T_c$)")
    for k in range(chip_count + 1):
        ax.axvline(k, color="0.65", lw=0.65, ls=":", zorder=0)
    for k, bit in enumerate(CHIPS_0000):
        ax.text(k + 0.5, 1.12, bit, ha="center", va="center", fontsize=8,
                color="#333333")
    ax.text(0.01, 0.97, "One vertical grid interval = one chip; digits above show chip values",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9, pad=3))
    ax.set_xlim(0, chip_count)
    ax.set_ylim(-1.28, 1.32)
    ax.set_ylabel("I/Q amplitude")
    ax.set_xlabel(r"Time (chip periods, $T_c$; here $T_c=0.5\,\mu$s at 2 Mchip/s)")
    ax.set_title("The I and Q components change smoothly; Q is offset by one chip period")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="lower right", ncol=2, framealpha=0.95)
    ax.set_xticks(np.arange(0, chip_count + 1, 2))

    # 3) Phase trajectory: this makes it explicit that there is no phase jump.
    ax = axes[2]
    ax.plot(t_chip, phase_deg, color="#8b5aa6", lw=2.0)
    for k in range(chip_count + 1):
        ax.axvline(k, color="0.72", lw=0.55, ls=":", zorder=0)
    ax.set_xlim(0, chip_count)
    ax.set_ylabel("Unwrapped phase (°)")
    ax.set_xlabel(r"Time (chip periods, $T_c$)")
    ax.set_title("The complex-envelope phase follows a continuous path through chip transitions")
    ax.grid(True, alpha=0.22)
    ax.set_xticks(np.arange(0, chip_count + 1, 2))

    # 4) The combined complex-envelope magnitude.
    ax = axes[3]
    ax.plot(t_chip, envelope, color="#478b54", lw=2.2,
            label=r"Envelope $\sqrt{I^2+Q^2}$")
    ax.axhline(1.0, color="0.35", lw=0.9, ls="--", label="Unit envelope")
    ax.set_xlim(0, chip_count)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Envelope")
    ax.set_xlabel(r"Time (chip periods, $T_c$)")
    ax.set_xticks(np.arange(0, chip_count + 1, 2))
    ax.set_title("The combined envelope stays constant after the initial ramp")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.suptitle(
        "Half-sine chip shaping in 2.4 GHz IEEE 802.15.4 O-QPSK\n"
        "The middle panels show the I/Q components and phase for all 32 chips of the 0000 DSSS data symbol",
        fontsize=15,
    )
    return fig


def open_image(path):
    image_path = str(path.resolve())
    if sys.platform == "darwin":
        subprocess.Popen(["open", image_path])
    elif os.name == "nt":
        os.startfile(image_path)
    else:
        subprocess.Popen(["xdg-open", image_path])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dsss_half_sine_chip_shaping.png",
                        help="output PNG path")
    parser.add_argument("--no-open", action="store_true",
                        help="save the PNG without opening it")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output.resolve()}")
    if not args.no_open:
        open_image(output)


if __name__ == "__main__":
    main()
