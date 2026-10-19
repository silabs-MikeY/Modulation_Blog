#!/usr/bin/env python3
"""Compare OQPSK spectra at two symbol rates around a 2.45 GHz carrier.

The same seeded 4096-bit stream is used for both rates. Each OQPSK symbol
represents two bits, and the Q branch is delayed by half a symbol. Rectangular
symbol pulses are used here to isolate the effect of symbol rate; this is an
illustration, not the half-sine chip shaping used by 2.4 GHz IEEE 802.15.4.

The signal is simulated at complex baseband, then its spectrum is shown with
the frequency axis shifted to the RF carrier. This avoids sampling the 2.45 GHz
carrier directly while preserving the spectrum around the positive-frequency
carrier.

Run:
    python3 dsss_oqpsk_symbol_rate_spectrum.py

Optional:
    python3 dsss_oqpsk_symbol_rate_spectrum.py --low-rate 250000 --high-rate 1000000
    python3 dsss_oqpsk_symbol_rate_spectrum.py --output figures/rate_compare.png --no-open
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-oqpsk-rate-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_oqpsk(bits, samples_per_symbol):
    """Create a unit-average-power, rectangular-pulse complex OQPSK signal."""
    if len(bits) % 2:
        raise ValueError("OQPSK requires an even number of input bits.")
    if samples_per_symbol < 2 or samples_per_symbol % 2:
        raise ValueError("samples_per_symbol must be an even integer of at least 2.")

    # Map 0/1 to -1/+1. Each branch carries one bit per OQPSK symbol.
    i_symbols = 2.0 * bits[0::2] - 1.0
    q_symbols = 2.0 * bits[1::2] - 1.0
    i_branch = np.repeat(i_symbols, samples_per_symbol)

    # Delay the Q branch by half a symbol. The initial value is extended to
    # provide the branch's value before the first delayed transition.
    q_unshifted = np.repeat(q_symbols, samples_per_symbol)
    half_symbol = samples_per_symbol // 2
    q_branch = np.concatenate((np.full(half_symbol, q_symbols[0]), q_unshifted))
    q_branch = q_branch[:len(i_branch)]

    return (i_branch + 1j * q_branch) / np.sqrt(2.0)


def welch_psd(signal, sample_rate, samples_per_symbol, segment_symbols=256):
    """Estimate a two-sided PSD with a Hann-windowed Welch average."""
    segment_length = min(segment_symbols * samples_per_symbol, len(signal))
    hop = max(1, segment_length // 2)
    n_fft = 4 * segment_length
    window = np.hanning(segment_length)
    scale = sample_rate * np.sum(window ** 2)
    power = np.zeros(n_fft, dtype=float)
    count = 0

    for start in range(0, len(signal) - segment_length + 1, hop):
        block = signal[start:start + segment_length]
        transformed = np.fft.fft(block * window, n=n_fft)
        power += np.abs(transformed) ** 2 / scale
        count += 1

    if count == 0:
        raise ValueError("Signal is too short for the requested PSD estimate.")

    power /= count
    offsets_hz = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    offsets_hz = np.fft.fftshift(offsets_hz)
    power = np.fft.fftshift(power)

    return offsets_hz, power


def plot_comparison(args):
    rng = np.random.default_rng(args.seed)
    bits = rng.integers(0, 2, size=args.bits, dtype=np.int8).astype(float)
    samples_per_symbol = args.samples_per_symbol
    symbols_per_branch = args.bits // 2

    rates = [args.low_rate, args.high_rate]
    colors = ["#2878b5", "#d14b40"]
    labels = ["Lower symbol rate", "Higher symbol rate"]

    fig, ax = plt.subplots(figsize=(12, 6.7), constrained_layout=True)
    max_rate = max(rates)
    spectra = []
    for rate, color, label in zip(rates, colors, labels):
        signal = make_oqpsk(bits, samples_per_symbol)
        sample_rate = rate * samples_per_symbol
        offsets_hz, power = welch_psd(signal, sample_rate, samples_per_symbol)
        spectra.append((rate, color, label, offsets_hz, power))

    # Both waveforms have the same average power. Reference both PSD curves to
    # one common peak so a wider spectrum also has a lower peak density.
    common_peak = max(float(np.max(power)) for _, _, _, _, power in spectra)
    for rate, color, label, offsets_hz, power in spectra:
        power_db = 10.0 * np.log10(np.maximum(power / common_peak, 1e-12))

        # Plot the carrier-centered positive-frequency RF spectrum in MHz.
        offsets_mhz = offsets_hz / 1e6
        cycles_per_symbol = args.carrier_hz / rate
        bit_rate = 2.0 * rate
        ax.plot(
            args.carrier_hz / 1e9 + offsets_mhz / 1000.0,
            power_db,
            color=color,
            linewidth=1.7,
            label=(f"{label}: {rate / 1e3:g} ksymbol/s "
                   f"({bit_rate / 1e3:g} kbit/s, "
                   f"{cycles_per_symbol:,.0f} carrier cycles/symbol)"),
        )

    # The sampling rate is chosen relative to the faster symbol rate, and the
    # plotted span makes the bandwidth change visible without hiding the lobes.
    span_mhz = max_rate / 1e6 * 5.0
    carrier_ghz = args.carrier_hz / 1e9
    ax.set_xlim(carrier_ghz - span_mhz / 1000.0,
                carrier_ghz + span_mhz / 1000.0)
    ax.set_ylim(-85, 5)
    ax.set_xlabel("RF frequency (GHz), centered on 2.45 GHz")
    ax.set_ylabel("Relative power spectral density (dB; common peak reference)")
    ax.set_title(
        f"OQPSK output spectrum: same {args.bits:,}-bit stream, different symbol rates\n"
        "Rectangular pulses for illustration; Q branch offset by half a symbol"
    )
    ax.axvline(carrier_ghz, color="0.35", linestyle=":", linewidth=1.0,
               label="2.45 GHz carrier")
    ax.grid(True, alpha=0.24)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.text(
        0.5, -0.01,
        "Complex-baseband simulation shifted to the RF carrier; no 2.45 GHz time-domain sampling.",
        ha="center", fontsize=9, color="0.35",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output.resolve()}")
    if not args.no_open:
        open_image(output)


def open_image(path):
    """Open the saved PNG with the platform's default image viewer."""
    image_path = str(path.resolve())
    if sys.platform == "darwin":
        subprocess.Popen(["open", image_path])
    elif os.name == "nt":
        os.startfile(image_path)
    else:
        subprocess.Popen(["xdg-open", image_path])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, default=4096,
                        help="number of data bits (default: 4096; must be even)")
    parser.add_argument("--low-rate", type=float, default=250_000,
                        help="lower OQPSK symbol rate in symbols/s (default: 250000)")
    parser.add_argument("--high-rate", type=float, default=1_000_000,
                        help="higher OQPSK symbol rate in symbols/s (default: 1000000)")
    parser.add_argument("--carrier-hz", type=float, default=2.45e9,
                        help="RF carrier frequency in Hz (default: 2.45e9)")
    parser.add_argument("--samples-per-symbol", type=int, default=32,
                        help="baseband samples per symbol (default: 32, even)")
    parser.add_argument("--seed", type=int, default=7,
                        help="random stream seed (default: 7)")
    parser.add_argument("--output", default="dsss_oqpsk_symbol_rate_spectrum.png",
                        help="output PNG path")
    parser.add_argument("--no-open", action="store_true",
                        help="save the PNG without opening it")
    args = parser.parse_args()

    if args.bits <= 0 or args.bits % 2:
        parser.error("--bits must be a positive even number")
    if args.low_rate <= 0 or args.high_rate <= 0:
        parser.error("symbol rates must be positive")
    if args.low_rate == args.high_rate:
        parser.error("--low-rate and --high-rate must differ")
    if args.samples_per_symbol < 2 or args.samples_per_symbol % 2:
        parser.error("--samples-per-symbol must be an even integer of at least 2")
    if args.carrier_hz <= 0:
        parser.error("--carrier-hz must be positive")
    return args


if __name__ == "__main__":
    plot_comparison(parse_args())
