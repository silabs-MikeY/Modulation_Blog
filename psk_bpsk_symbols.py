"""Visualize the two BPSK symbols with Matplotlib.

Each row pairs a short carrier waveform with the BPSK constellation. BPSK
changes carrier phase by 180 degrees; its ideal amplitude envelope stays
constant.

Run with: python psk_bpsk_symbols.py
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    amplitude = 1.0
    carrier_frequency = 2.45e9  # Hz
    carrier_cycles = 1
    samples_per_cycle = 160
    time_s = np.linspace(0, carrier_cycles / carrier_frequency,
                         carrier_cycles * samples_per_cycle, endpoint=False)
    time_ns = time_s * 1e9
    # Use a sine reference so phase 0° starts at zero amplitude and rises.
    # The bit-to-phase mapping is a convention; here 0 -> 0° and 1 -> 180°.
    phase_values = (0, np.pi)
    labels = ("Bit 0  (phase 0°)", "Bit 1  (phase 180°)")
    colors = ("tab:blue", "tab:orange")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    fig.suptitle("BPSK: two symbols, two phases, constant envelope", fontsize=15)

    constellation = np.array([[amplitude, 0], [-amplitude, 0]])
    for row, (phase, label, color) in enumerate(zip(phase_values, labels, colors)):
        signal = amplitude * np.sin(2 * np.pi * carrier_frequency * time_s + phase)
        wave_ax = axes[row, 0]
        wave_ax.plot(time_ns, signal, color=color, linewidth=1.8)
        wave_ax.axhline(amplitude, color="0.55", linestyle="--", linewidth=1)
        wave_ax.axhline(-amplitude, color="0.55", linestyle="--", linewidth=1)
        wave_ax.set_title(label, loc="left")
        wave_ax.set_ylabel("Signal amplitude")
        wave_ax.set_xlabel("Time (ns)")
        wave_ax.text(0.98, 0.96, f"Carrier: {carrier_frequency / 1e9:.2f} GHz",
                     transform=wave_ax.transAxes, ha="right", va="top",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                           "pad": 2})
        wave_ax.set_ylim(-1.25 * amplitude, 1.25 * amplitude)
        wave_ax.grid(True, alpha=0.25)

        constellation_ax = axes[row, 1]
        constellation_ax.axhline(0, color="0.75", linewidth=1)
        constellation_ax.axvline(0, color="0.75", linewidth=1)
        constellation_ax.scatter(constellation[:, 0], constellation[:, 1],
                                 color="0.75", s=65, zorder=2)
        constellation_ax.scatter([amplitude * np.cos(phase)],
                                 [amplitude * np.sin(phase)], color=color,
                                 edgecolor="black", s=130, zorder=3)
        constellation_ax.set_title("Phase constellation")
        constellation_ax.set_xlabel("cos")
        constellation_ax.set_ylabel("sin")
        constellation_ax.set_xlim(-1.35 * amplitude, 1.35 * amplitude)
        constellation_ax.set_ylim(-1.2 * amplitude, 1.2 * amplitude)
        constellation_ax.set_aspect("equal", adjustable="box")
        constellation_ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # An ideal, continuous sine-wave carrier has a single spectral line.
    spectrum_fig, spectrum_ax = plt.subplots(figsize=(9, 4.5), num="BPSK waveform spectrum")
    spectrum_ax.vlines(carrier_frequency / 1e9, 0, amplitude, color="tab:purple",
                       linewidth=2.5, label="Carrier: 2.45 GHz")
    spectrum_ax.set_title("Ideal carrier spectrum")
    spectrum_ax.set_xlabel("Frequency (GHz)")
    spectrum_ax.set_ylabel("Magnitude (relative amplitude)")
    spectrum_ax.set_xlim(0, 8)
    spectrum_ax.set_ylim(0, 1.15 * amplitude)
    spectrum_ax.grid(True, alpha=0.25)
    spectrum_ax.legend()
    spectrum_fig.tight_layout()

    # Show how a short bit sequence selects the BPSK phase, one cycle per bit.
    bits = np.array([0, 1, 1, 1, 0, 1])
    samples_per_bit = samples_per_cycle
    sample_count = len(bits) * samples_per_bit
    bit_time = np.arange(sample_count) / samples_per_bit
    bit_phases = np.repeat(np.where(bits == 0, 0.0, np.pi), samples_per_bit)
    bit_waveform = amplitude * np.sin(
        2 * np.pi * (bit_time % 1) + bit_phases
    )
    stream_fig, (wave_stream_ax, bits_ax) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, num="BPSK bit stream and waveform",
        gridspec_kw={"height_ratios": [2, 1]}
    )
    stream_fig.suptitle("Mapping shown: bit 0 → 0°, bit 1 → 180°")
    wave_stream_ax.plot(bit_time, bit_waveform, color="tab:blue", linewidth=1.5)
    wave_stream_ax.set_title("BPSK waveform: one carrier cycle per bit")
    wave_stream_ax.set_ylabel("Signal amplitude")
    wave_stream_ax.set_ylim(-1.25 * amplitude, 1.25 * amplitude)
    wave_stream_ax.grid(True, alpha=0.25)
    bits_ax.step(np.arange(len(bits) + 1), np.r_[bits, bits[-1]], where="post",
                 color="tab:orange", linewidth=2)
    bits_ax.set_title("Sample bit stream")
    bits_ax.set_ylabel("Bit value")
    bits_ax.set_xlabel("Bit interval")
    bits_ax.set_xlim(0, len(bits))
    bits_ax.set_ylim(-0.25, 1.25)
    bits_ax.set_yticks([0, 1])
    bits_ax.set_xticks(np.arange(len(bits) + 1))
    bits_ax.grid(True, alpha=0.25)
    for boundary in np.arange(1, len(bits)):
        wave_stream_ax.axvline(boundary, color="0.6", linestyle=":", linewidth=1)
        bits_ax.axvline(boundary, color="0.6", linestyle=":", linewidth=1)
    stream_fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Spectrum of the complete, finite 011101 burst. No tapering window is
    # applied, so the abrupt phase changes and burst edges remain visible.
    fft_size = 65536
    burst_spectrum = np.abs(np.fft.rfft(bit_waveform, n=fft_size))
    burst_spectrum_db = 20 * np.log10(
        np.maximum(burst_spectrum / np.max(burst_spectrum), 1e-4)
    )
    sample_rate = carrier_frequency * samples_per_bit
    burst_frequency_ghz = np.fft.rfftfreq(fft_size, d=1 / sample_rate) / 1e9
    burst_fig, burst_ax = plt.subplots(
        figsize=(10, 5), num="Spectrum of 011101 BPSK waveform"
    )
    burst_ax.plot(burst_frequency_ghz, burst_spectrum_db,
                  color="tab:purple", linewidth=1.3)
    burst_ax.axvline(carrier_frequency / 1e9, color="0.4", linestyle="--",
                     linewidth=1, label="Carrier: 2.45 GHz")
    burst_ax.set_title("Spectrum of finite 011101 BPSK burst")
    burst_ax.set_xlabel("Frequency (GHz)")
    burst_ax.set_ylabel("Magnitude (dB, relative to peak)")
    burst_ax.set_xlim(0, 30)
    burst_ax.set_ylim(-80, 3)
    burst_ax.grid(True, alpha=0.25)
    burst_ax.legend()
    burst_fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
