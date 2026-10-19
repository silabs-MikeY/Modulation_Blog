#!/usr/bin/env python3
"""Compare linear gain and PA spectral regrowth for a 2.45 GHz BPSK stream.

The bit pattern and 10-cycle raised-cosine transitions match
psk_bpsk_slow_stream.py. Its phase reversals cross zero amplitude.

Run: python3 rf_amplification_gain.py
Optional: python3 rf_amplification_gain.py --output amplifier_gain.png
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-qpsk-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="rf_amplification_gain.png",
                        help="Output image path (default: %(default)s)")
    parser.add_argument("--gain", type=float, default=3.0,
                        help="Voltage gain (default: %(default)s)")
    parser.add_argument("--carrier-ghz", type=float, default=2.45,
                        help="Carrier frequency in GHz (default: %(default)s)")
    parser.add_argument("--no-open", action="store_true",
                        help="Save the PNG without opening it")
    args = parser.parse_args()
    if args.gain <= 0:
        parser.error("--gain must be positive")
    if args.carrier_ghz <= 0:
        parser.error("--carrier-ghz must be positive")

    gain = args.gain
    input_amplitude = 1.0

    carrier_hz = args.carrier_ghz * 1e9
    bits = np.array([0, 1, 1, 1, 0, 1])
    transition_cycles = 10
    hold_cycles = 10
    cycles_per_symbol = transition_cycles + hold_cycles
    samples_per_cycle = 240
    samples_per_symbol = cycles_per_symbol * samples_per_cycle
    sample_rate = carrier_hz * samples_per_cycle
    time_cycles = np.arange(len(bits) * samples_per_symbol) / samples_per_cycle
    time_symbols = time_cycles / cycles_per_symbol
    carrier = np.sin(2 * np.pi * time_cycles)
    symbol_levels = np.where(bits == 0, 1.0, -1.0)
    signed_envelope = np.empty(time_cycles.size)
    within_symbol_cycles = np.arange(samples_per_symbol) / samples_per_cycle
    for index, target in enumerate(symbol_levels):
        start = index * samples_per_symbol
        stop = start + samples_per_symbol
        if index == 0 or target == symbol_levels[index - 1]:
            signed_envelope[start:stop] = target
        else:
            previous = symbol_levels[index - 1]
            blend = 0.5 * (1 - np.cos(
                np.pi * np.minimum(within_symbol_cycles / transition_cycles, 1)
            ))
            signed_envelope[start:stop] = previous + (target - previous) * blend
    input_wave = signed_envelope * carrier
    output_wave = gain * input_wave

    # Generic memoryless soft-compression AM/AM model (Rapp-style). These
    # normalized parameters are illustrative, not measured EFR32 data.
    saturation_output = 2.4
    smoothness = 2.0

    def pa_amam(input_envelope):
        return (gain * input_envelope /
                (1.0 + (gain * input_envelope / saturation_output) ** (2.0 * smoothness))
                ** (1.0 / (2.0 * smoothness)))

    pa_output_wave = np.sign(signed_envelope) * pa_amam(np.abs(signed_envelope)) * carrier

    # Produce three separate figures: the original linear-gain illustration,
    # the PA transfer curve, and the nonlinear waveform/spectrum comparison.
    fig_linear, (wave_ax, transfer_ax, linear_spectrum_ax) = plt.subplots(
        3, 1, figsize=(12, 13), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.4, 0.85, 1]},
    )
    fig_linear.suptitle(f"Linear gain for a {args.carrier_ghz:g} GHz BPSK stream (G = {gain:g})", fontsize=18)

    fig_amam, amam_ax = plt.subplots(1, 1, figsize=(9, 7), constrained_layout=True)
    fig_amam.suptitle("Illustrative EFR32 Series 2-style PA AM/AM response", fontsize=16)

    fig_nonlinear, (pa_wave_ax, spectrum_ax) = plt.subplots(
        2, 1, figsize=(12, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 1]},
    )
    fig_nonlinear.suptitle(f"PA spectral regrowth for the same {args.carrier_ghz:g} GHz BPSK stream", fontsize=18)

    input_envelope = np.abs(signed_envelope)

    def draw_input(ax):
        ax.plot(time_symbols, input_wave, color="tab:blue", lw=1.1,
                label="BPSK input RF waveform", zorder=5)
        ax.plot(time_symbols, input_envelope, color="tab:orange", ls="--", lw=1,
                alpha=0.8, label="Input envelope")
        ax.plot(time_symbols, -input_envelope, color="tab:orange", ls="--", lw=1,
                alpha=0.8)
        ax.set_title("Carrier response: 10-cycle transition, then 10-cycle hold")
        ax.set_ylabel("Signal amplitude")
        ax.set_xlim(0, len(bits))
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks(np.arange(len(bits) + 1))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        for boundary in range(1, len(bits)):
            ax.axvline(boundary, color="0.6", ls=":", lw=0.8)
            if bits[boundary] != bits[boundary - 1]:
                ax.axvspan(boundary, boundary + transition_cycles / cycles_per_symbol,
                           color="tab:orange", alpha=0.10)

    wave_ax.plot(time_symbols, output_wave, color="#d62728", lw=1.0,
                 label=f"Ideal {gain:g}× output RF waveform", zorder=2)
    wave_ax.plot(time_symbols, input_wave, color="tab:blue", lw=1.1,
                 label="BPSK input RF waveform", zorder=3)
    wave_ax.plot(time_symbols, gain * input_envelope, color="#d62728", ls="--",
                 lw=1.1, label="Ideal output envelope")
    wave_ax.plot(time_symbols, -gain * input_envelope, color="#d62728", ls="--", lw=1.1)
    wave_ax.plot(time_symbols, input_envelope, color="tab:orange", ls="--",
                 lw=1.0, label="Input envelope")
    wave_ax.plot(time_symbols, -input_envelope, color="tab:orange", ls="--", lw=1.0)
    wave_ax.set_title("BPSK input and ideal linear output over the full stream")
    wave_ax.set_xlabel("Time (symbol intervals)")
    wave_ax.set_ylabel("RF voltage (normalized)")
    wave_ax.set_xlim(0, len(bits))
    wave_ax.set_ylim(-gain * 1.15, gain * 1.15)
    wave_ax.set_xticks(np.arange(len(bits) + 1))
    wave_ax.grid(True, alpha=0.25)
    wave_ax.legend(loc="upper right", framealpha=0.95)
    for boundary in range(1, len(bits)):
        wave_ax.axvline(boundary, color="0.6", ls=":", lw=0.8)
        if bits[boundary] != bits[boundary - 1]:
            wave_ax.axvspan(boundary, boundary + transition_cycles / cycles_per_symbol,
                            color="tab:orange", alpha=0.10)

    input_voltage = np.linspace(-input_amplitude, input_amplitude, 300)
    output_voltage = gain * input_voltage
    transfer_ax.plot(input_voltage, output_voltage, color="#d62728", lw=2.5,
                     label=rf"$V_{{out}} = {gain:g}V_{{in}}$")
    transfer_ax.scatter([-1, 0, 1], [-gain, 0, gain], color="#d62728",
                        edgecolor="black", linewidth=0.8, zorder=3)
    transfer_ax.axhline(0, color="0.45", lw=0.8)
    transfer_ax.axvline(0, color="0.45", lw=0.8)
    transfer_ax.set_title("Linear voltage gain")
    transfer_ax.set_xlabel("Input voltage (normalized)")
    transfer_ax.set_ylabel("Output voltage (normalized)")
    transfer_ax.set_xlim(-1.15, 1.15)
    transfer_ax.set_ylim(-gain * 1.15, gain * 1.15)
    transfer_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    transfer_ax.set_yticks(np.linspace(-gain, gain, 7))
    transfer_ax.grid(True, alpha=0.28)
    transfer_ax.legend(loc="upper left", framealpha=0.95)

    curve_input = np.linspace(0, input_amplitude * 1.15, 400)
    pa_curve_output = pa_amam(curve_input)
    amam_ax.plot(curve_input, pa_curve_output, color="#d62728", lw=2.5,
                 label="Illustrative PA AM/AM curve")
    amam_ax.plot(curve_input, gain * curve_input, color="#2878b5", lw=1.8,
                 ls="--", label=rf"Linear reference ($G={gain:g}$)")
    amam_ax.scatter([input_amplitude], [pa_amam(input_amplitude)], color="#d62728",
                    edgecolor="black", linewidth=0.8, zorder=3)
    amam_ax.set_title("EFR32 Series 2-style PA AM/AM response (illustrative)")
    amam_ax.set_xlabel("Input envelope amplitude (normalized)")
    amam_ax.set_ylabel("Output envelope amplitude (normalized)")
    amam_ax.text(0.99, 0.04,
                 "Generic soft-compression model; not measured EFR32 data",
                 transform=amam_ax.transAxes, ha="right", va="bottom", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
    amam_ax.set_xlim(0, input_amplitude * 1.15)
    amam_ax.set_ylim(0, max(pa_curve_output.max(), gain * input_amplitude * 1.15) * 1.08)
    amam_ax.grid(True, alpha=0.28)
    amam_ax.legend(loc="upper left", framealpha=0.95)

    draw_input(pa_wave_ax)
    pa_envelope = pa_amam(input_envelope)
    pa_wave_ax.plot(time_symbols, pa_output_wave, color="#d62728", lw=1.1,
                    label="Compressed PA RF waveform", zorder=3)
    pa_wave_ax.plot(time_symbols, pa_envelope, color="#d62728", ls="--", lw=1.1,
                    label="Compressed PA envelope")
    pa_wave_ax.plot(time_symbols, -pa_envelope, color="#d62728", ls="--", lw=1.1)
    pa_wave_ax.plot(time_symbols, gain * input_envelope, color="0.45", ls=":", lw=1,
                    label="Ideal linear envelope")
    pa_wave_ax.plot(time_symbols, -gain * input_envelope, color="0.45", ls=":", lw=1)
    pa_wave_ax.set_title("BPSK input and compressed PA output over the full stream")
    pa_wave_ax.set_xlabel("Time (symbol intervals)")
    pa_wave_ax.set_ylabel("RF voltage (normalized)")
    pa_wave_ax.set_xlim(0, len(bits))
    pa_wave_ax.set_ylim(-gain * 1.15, gain * 1.15)
    pa_wave_ax.set_xticks(np.arange(len(bits) + 1))
    pa_wave_ax.grid(True, alpha=0.25)
    pa_wave_ax.legend(loc="upper right", framealpha=0.95)
    for boundary in range(1, len(bits)):
        pa_wave_ax.axvline(boundary, color="0.6", ls=":", lw=0.8)
        if bits[boundary] != bits[boundary - 1]:
            pa_wave_ax.axvspan(boundary, boundary + transition_cycles / cycles_per_symbol,
                               color="tab:orange", alpha=0.10)

    # Analyze the full waveform plotted above, with the same window for every
    # trace. The transition changes its RF envelope and the PA distorts it.
    carrier_ghz = args.carrier_ghz
    fft_size = 262144
    window = np.hanning(input_wave.size)
    rf_frequency_ghz = np.fft.rfftfreq(fft_size, d=1 / sample_rate) / 1e9
    def spectrum(signal):
        return np.abs(np.fft.rfft(signal * window, n=fft_size)) / window.sum()
    input_spectrum = spectrum(input_wave)
    output_spectrum = spectrum(output_wave)
    pa_spectrum = spectrum(pa_output_wave)
    inband = np.abs(rf_frequency_ghz - carrier_ghz) < 0.1
    reference = output_spectrum[inband].max()
    pa_reference = pa_spectrum[inband].max()
    input_dbc = 20 * np.log10(np.maximum(input_spectrum / reference, 1e-6))
    output_dbc = 20 * np.log10(np.maximum(output_spectrum / reference, 1e-6))
    # Align the PA inband peak with the ideal output to compare spectral shape.
    pa_dbc = 20 * np.log10(np.maximum(pa_spectrum / pa_reference, 1e-6))
    # Match the nonlinear panel's window, FFT, dBc reference, and frequency view.
    linear_spectrum_ax.plot(rf_frequency_ghz, output_dbc,
                            color="#d62728", lw=1.5,
                            label=f"Ideal {gain:g}× output", zorder=2)
    linear_spectrum_ax.plot(rf_frequency_ghz, input_dbc,
                            color="tab:blue", lw=1.5, label="BPSK input", zorder=4)
    linear_spectrum_ax.axvline(carrier_ghz, color="0.4", ls=":", lw=1,
                               label=f"Carrier: {carrier_ghz:g} GHz")
    linear_spectrum_ax.set_title("Windowed carrier zoom: BPSK input and ideal linear output")
    linear_spectrum_ax.set_xlabel("RF frequency (GHz)")
    linear_spectrum_ax.set_ylabel("Magnitude (dBc)")
    linear_spectrum_ax.set_xlim(carrier_ghz - 0.65, carrier_ghz + 0.65)
    linear_spectrum_ax.set_ylim(-95, 5)
    linear_spectrum_ax.grid(True, alpha=0.28)
    linear_spectrum_ax.legend(loc="upper right", framealpha=0.95)

    spectrum_ax.plot(rf_frequency_ghz, output_dbc, color="#888888", lw=1.3, ls="--",
                     label="Ideal linear output", zorder=2)
    spectrum_ax.plot(rf_frequency_ghz, pa_dbc, color="#d62728", lw=1.6,
                     label="Compressed PA output (inband peak aligned)", zorder=4)
    spectrum_ax.plot(rf_frequency_ghz, input_dbc, color="#2878b5", lw=1.4,
                     label="BPSK input", zorder=5)
    spectrum_ax.axvline(carrier_ghz, color="0.4", ls=":", lw=1,
                        label=f"Carrier: {carrier_ghz:g} GHz")
    spectrum_ax.set_title("Windowed carrier zoom: PA compression raises spectral shoulders")
    spectrum_ax.set_xlabel("RF frequency (GHz)")
    spectrum_ax.set_ylabel("Magnitude (dBc)")
    spectrum_ax.set_xlim(carrier_ghz - 0.65, carrier_ghz + 0.65)
    spectrum_ax.set_ylim(-95, 5)
    spectrum_ax.text(0.01, 0.04,
                     "PA trace aligned at its inband peak to compare spectral shape",
                     transform=spectrum_ax.transAxes, fontsize=9,
                     bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
    spectrum_ax.grid(True, alpha=0.28)
    spectrum_ax.legend(loc="upper right", framealpha=0.95)

    output = Path(args.output)
    amam_output = output.with_name(f"{output.stem}_amam{output.suffix}")
    nonlinear_output = output.with_name(f"{output.stem}_nonlinear{output.suffix}")
    figures = [(fig_linear, output), (fig_amam, amam_output),
               (fig_nonlinear, nonlinear_output)]
    for figure, image_path in figures:
        figure.savefig(image_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {image_path.resolve()}")

    if not args.no_open:
        for _, image_path in figures:
            image_path = str(image_path.resolve())
            if sys.platform == "darwin":
                subprocess.Popen(["open", image_path])
            elif os.name == "nt":
                os.startfile(image_path)
            else:
                subprocess.Popen(["xdg-open", image_path])


if __name__ == "__main__":
    main()
