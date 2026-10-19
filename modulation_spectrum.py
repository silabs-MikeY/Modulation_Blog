"""Shared helpers for illustrative QPSK/OQPSK output spectra."""

import numpy as np


def qpsk_envelope(symbols, state_bits, cycles_per_symbol=6, samples_per_cycle=500):
    """Return a raised-cosine Cartesian QPSK envelope for the given bit pairs."""
    degree_for_bits = {bits: degree for degree, bits in state_bits.items()}
    angles = np.deg2rad([degree_for_bits[bits] for bits in symbols])
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    samples_per_symbol = int(round(cycles_per_symbol * samples_per_cycle))
    t = np.arange(len(symbols) * samples_per_symbol + 1) / samples_per_cycle
    x_component = np.full_like(t, points[0, 0])
    y_component = np.full_like(t, points[0, 1])
    progress = np.arange(samples_per_symbol) / samples_per_symbol
    amount = 0.5 - 0.5 * np.cos(np.pi * progress)

    for idx in range(1, len(symbols)):
        start = idx * samples_per_symbol
        end = start + samples_per_symbol
        source, target = points[idx - 1], points[idx]
        x_component[start:end] = source[0] + (target[0] - source[0]) * amount
        y_component[start:end] = source[1] + (target[1] - source[1]) * amount

    x_component[-1], y_component[-1] = points[-1]
    return t, x_component + 1j * y_component


def oqpsk_components(symbols, state_bits, cycles_per_symbol=6, samples_per_cycle=500):
    """Return time and smoothly staggered bit1/bit0 OQPSK branch levels."""
    degree_for_bits = {bits: degree for degree, bits in state_bits.items()}
    angles = np.deg2rad([degree_for_bits[bits] for bits in symbols])
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    samples_per_symbol = int(round(cycles_per_symbol * samples_per_cycle))
    half_symbol = samples_per_symbol // 2
    t = np.arange(len(symbols) * samples_per_symbol + 1) / samples_per_cycle
    bit1 = np.full_like(t, points[0, 0])
    bit0 = np.full_like(t, points[0, 1])
    progress = np.arange(samples_per_symbol) / samples_per_symbol
    amount = 0.5 - 0.5 * np.cos(np.pi * progress)

    # Each OQPSK branch uses the same full-symbol raised-cosine transition as
    # QPSK. The bit0 branch is delayed by half a symbol.
    for idx in range(1, len(symbols)):
        source, target = points[idx - 1], points[idx]
        start1 = idx * samples_per_symbol
        end1 = min(start1 + samples_per_symbol, len(t))
        count1 = end1 - start1
        bit1[start1:end1] = source[0] + (target[0] - source[0]) * amount[:count1]

        start0 = idx * samples_per_symbol + half_symbol
        end0 = min(start0 + samples_per_symbol, len(t))
        count0 = end0 - start0
        bit0[start0:end0] = source[1] + (target[1] - source[1]) * amount[:count0]

    bit1[-1] = points[-1, 0]
    return t, bit1, bit0


def pa_soft_compression(envelope, gain=3.0, saturation_output=2.4, smoothness=2.0):
    """Apply the illustrative memoryless AM/AM curve used by rf_amplification_gain.py."""
    input_magnitude = np.abs(envelope)
    output_magnitude = (gain * input_magnitude /
                        (1.0 + (gain * input_magnitude / saturation_output)
                         ** (2.0 * smoothness))
                        ** (1.0 / (2.0 * smoothness)))
    scale = np.divide(output_magnitude, input_magnitude,
                      out=np.zeros_like(output_magnitude), where=input_magnitude > 0)
    return envelope * scale


def spectrum_db(envelope, cycles_per_symbol=6, samples_per_cycle=500,
                reference_bandwidth=1.0, carrier_ghz=2.45):
    """Return a Welch-averaged spectrum on an absolute RF-frequency axis in GHz."""
    samples_per_symbol = int(round(cycles_per_symbol * samples_per_cycle))
    segment_symbols = 128
    segment_length = samples_per_symbol * segment_symbols
    hop = segment_length // 2
    n_fft = 4 * segment_length
    window = np.hanning(segment_length)
    window_power = np.sum(window ** 2)
    power = np.zeros(n_fft)
    segment_count = 0
    for start in range(0, len(envelope) - segment_length + 1, hop):
        segment = envelope[start:start + segment_length]
        spectrum = np.fft.fftshift(np.fft.fft(segment * window, n=n_fft))
        power += np.abs(spectrum) ** 2 / window_power
        segment_count += 1
    power /= max(segment_count, 1)
    frequency = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / samples_per_symbol))
    inband = np.abs(frequency) <= reference_bandwidth
    normalizer = max(float(np.sum(power[inband])), np.finfo(float).tiny)
    power_db = 10.0 * np.log10(np.maximum(power / normalizer, 1e-14))
    symbol_rate_ghz = carrier_ghz / cycles_per_symbol
    return carrier_ghz + frequency * symbol_rate_ghz, power_db


def plot_spectra(ax, curves, cycles_per_symbol=6, samples_per_cycle=500,
                 reference_bandwidth=1.0, carrier_ghz=2.45):
    """Plot labeled (name, envelope, color, linestyle) curves on an axes."""
    for label, envelope, color, linestyle in curves:
        frequency, power_db = spectrum_db(
            envelope, cycles_per_symbol=cycles_per_symbol,
            reference_bandwidth=reference_bandwidth,
            carrier_ghz=carrier_ghz,
            samples_per_cycle=samples_per_cycle,
        )
        ax.plot(frequency, power_db, color=color, linestyle=linestyle,
                linewidth=1.5, label=label)

    symbol_rate_ghz = carrier_ghz / cycles_per_symbol
    ax.axvspan(carrier_ghz - reference_bandwidth * symbol_rate_ghz,
               carrier_ghz + reference_bandwidth * symbol_rate_ghz,
               color="#93c5fd", alpha=0.18, label="Reference in-band")
    ax.axvline(carrier_ghz, color="0.35", lw=0.9, ls=":", label="2.45 GHz carrier")
    ax.set_xlim(max(0.0, carrier_ghz - 4.0 * symbol_rate_ghz),
                carrier_ghz + 4.0 * symbol_rate_ghz)
    ax.set_ylim(-75, 5)
    ax.set_xlabel("Absolute frequency (GHz)")
    ax.set_ylabel("Relative power spectrum (dB)")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", ncol=2, fontsize=9, framealpha=0.95)
