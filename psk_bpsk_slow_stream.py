"""Plot a 011101 BPSK waveform with 10-cycle transitions and its spectrum.

Run with: python psk_bpsk_slow_stream.py

The example maps bit 0 to +1 and bit 1 to -1. Bits are latched at symbol
boundaries. When adjacent bits differ, the carrier transitions over the
first 10 cycles of the 20-cycle symbol interval, then holds the new symbol.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    carrier_frequency = 2.45e9
    bits = np.array([0, 1, 1, 1, 0, 1])
    transition_cycles = 10
    hold_cycles = 10
    cycles_per_symbol = transition_cycles + hold_cycles
    samples_per_cycle = 240
    samples_per_symbol = cycles_per_symbol * samples_per_cycle
    sample_count = len(bits) * samples_per_symbol
    sample_rate = carrier_frequency * samples_per_cycle

    time_cycles = np.arange(sample_count) / samples_per_cycle
    carrier = np.sin(2 * np.pi * time_cycles)
    symbol_levels = np.where(bits == 0, 1.0, -1.0)
    carrier_multiplier = np.empty(sample_count)
    within_symbol_cycles = np.arange(samples_per_symbol) / samples_per_cycle

    for symbol_index, target in enumerate(symbol_levels):
        start = symbol_index * samples_per_symbol
        stop = start + samples_per_symbol
        if symbol_index == 0 or target == symbol_levels[symbol_index - 1]:
            carrier_multiplier[start:stop] = target
        else:
            previous = symbol_levels[symbol_index - 1]
            blend = 0.5 * (
                1 - np.cos(np.pi * np.minimum(within_symbol_cycles / transition_cycles, 1))
            )
            carrier_multiplier[start:stop] = previous + (target - previous) * blend

    waveform = carrier_multiplier * carrier
    time_in_symbols = time_cycles / cycles_per_symbol

    waveform_fig, (wave_ax, bits_ax) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        num="011101 BPSK waveform with slow transitions",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    waveform_fig.suptitle(
        "BPSK bit stream 011101 — 10 carrier cycles per symbol transition\n"
        f"Carrier frequency: {carrier_frequency / 1e9:.2f} GHz"
    )

    wave_ax.plot(time_in_symbols, waveform, color="tab:blue", linewidth=1.1)
    envelope = np.abs(carrier_multiplier)
    wave_ax.plot(time_in_symbols, envelope, color="tab:orange",
                 linestyle="--", linewidth=1, alpha=0.8, label="Envelope")
    wave_ax.plot(time_in_symbols, -envelope, color="tab:orange",
                 linestyle="--", linewidth=1, alpha=0.8)
    wave_ax.set_title("Carrier response: 10-cycle transition, then 10-cycle hold")
    wave_ax.set_ylabel("Signal amplitude")
    wave_ax.set_ylim(-1.25, 1.25)
    wave_ax.grid(True, alpha=0.25)
    wave_ax.legend(loc="upper right")

    bits_ax.step(np.arange(len(bits) + 1), np.r_[bits, bits[-1]], where="post",
                 color="tab:purple", linewidth=2)
    bits_ax.set_title("Input bits (target symbols)")
    bits_ax.set_ylabel("Bit")
    bits_ax.set_xlabel("Symbol interval (bit latched at each left boundary)")
    bits_ax.set_xlim(0, len(bits))
    bits_ax.set_ylim(-0.25, 1.25)
    bits_ax.set_yticks([0, 1])
    bits_ax.set_xticks(np.arange(len(bits) + 1))
    bits_ax.grid(True, alpha=0.25)

    for boundary in range(1, len(bits)):
        wave_ax.axvline(boundary, color="0.6", linestyle=":", linewidth=0.8)
        bits_ax.axvline(boundary, color="0.6", linestyle=":", linewidth=0.8)
        if bits[boundary] != bits[boundary - 1]:
            wave_ax.axvspan(
                boundary, boundary + transition_cycles / cycles_per_symbol,
                color="tab:orange", alpha=0.10,
            )

    waveform_fig.tight_layout(rect=(0, 0, 1, 0.92))

    # FFT of the entire 011101 waveform; zero-padding smooths the display.
    fft_size = 65536
    spectrum = 2 * np.abs(np.fft.rfft(waveform, n=fft_size)) / len(waveform)
    spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-4))
    frequency_ghz = np.fft.rfftfreq(fft_size, d=1 / sample_rate) / 1e9

    spectrum_fig, spectrum_ax = plt.subplots(
        figsize=(10, 5), num="011101 BPSK waveform spectrum"
    )
    spectrum_ax.plot(frequency_ghz, spectrum_db, color="tab:green", linewidth=1.3)
    spectrum_ax.axvline(carrier_frequency / 1e9, color="0.4", linestyle="--",
                        linewidth=1, label="Carrier: 2.45 GHz")
    spectrum_ax.set_title("Spectrum of the entire 011101 waveform")
    spectrum_ax.set_xlabel("Frequency (GHz)")
    spectrum_ax.set_ylabel("Magnitude (dB relative to symbol amplitude)")
    spectrum_ax.set_xlim(0, 30)
    spectrum_ax.set_ylim(-80, 5)
    spectrum_ax.grid(True, alpha=0.25)
    spectrum_ax.legend()
    spectrum_fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
