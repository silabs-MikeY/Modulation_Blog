#!/usr/bin/env python3
"""Compare finite-edge and half-sine-shaped OQPSK spectra without DSSS.

A seeded bitstream directly updates alternating I/Q branches at 250 ksymbols/s
by default. The reference uses short, smooth branch transitions that follow
straight paths between constellation states and dip inward. The other sums
overlapping half-sine branch pulses and follows circular arcs. Both pass through
the same illustrative Rapp AM/AM PA model. The finite edge duration is an
explicit modeling choice, not an EFR32 transmitter specification.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mpl-oqpsk-shaping-spectrum-cache'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_oqpsk_pair(bits, samples_per_bit, edge_fraction=0.25):
    """Return finite-edge and half-sine OQPSK for the same bits.

    Bits update alternating branches, with the quadrature branch delayed by
    one bit interval (half an OQPSK symbol). In the shaped waveform each branch
    bit contributes a half-sine pulse lasting two bit intervals; adjacent I/Q
    pulses overlap in quadrature. The reference takes a straight path between
    adjacent states during a raised-cosine branch edge of ``edge_fraction`` bit
    intervals, holding each state otherwise. Both visit the same constellation
    state at the middle of each bit interval.
    """
    bit_count = len(bits)
    t_bits = (np.arange(bit_count * samples_per_bit) + 0.5) / samples_per_bit
    event_index = np.floor(t_bits).astype(np.int64) % bit_count
    tau = t_bits - np.floor(t_bits)
    current_is_i = (event_index % 2) == 0
    current_bit = bits[event_index]
    previous_bit = bits[(event_index - 1) % bit_count]

    # Match the project's Gray labels: 00/01/10/11 map to
    # 135/45/225/315 degrees, respectively.
    current = np.where(current_is_i, 2.0 * current_bit - 1.0,
                       1.0 - 2.0 * current_bit)
    previous = np.where(current_is_i, 1.0 - 2.0 * previous_bit,
                        2.0 * previous_bit - 1.0)

    sine = np.sin(np.pi * tau / 2.0)
    cosine = np.cos(np.pi * tau / 2.0)
    i_half_sine = np.where(current_is_i, current * sine, previous * cosine)
    q_half_sine = np.where(current_is_i, previous * cosine, current * sine)
    half_sine = i_half_sine + 1j * q_half_sine

    # At the middle of each bit interval both waveforms visit the same state.
    # A finite transition between states dips inside the unit circle whenever
    # one branch changes sign. Its timing is deliberately explicit: a full-bit
    # ramp would itself be a strong spectral-shaping filter.
    scale = 1.0 / np.sqrt(2.0)
    event_numbers = np.arange(bit_count)
    i_center = np.where(event_numbers % 2 == 0,
                        2.0 * bits[event_numbers] - 1.0,
                        2.0 * bits[(event_numbers - 1) % bit_count] - 1.0)
    q_center = np.where(event_numbers % 2 == 0,
                        1.0 - 2.0 * bits[(event_numbers - 1) % bit_count],
                        1.0 - 2.0 * bits[event_numbers])
    centers = (i_center + 1j * q_center) * scale
    before_center = tau < 0.5
    start = np.where(before_center, centers[(event_index - 1) % bit_count],
                     centers[event_index])
    stop = np.where(before_center, centers[event_index],
                    centers[(event_index + 1) % bit_count])
    distance_from_boundary = np.where(before_center, tau, tau - 1.0)
    progress = np.clip(0.5 + distance_from_boundary / edge_fraction, 0.0, 1.0)
    blend = 0.5 - 0.5 * np.cos(np.pi * progress)
    finite_edge = start + (stop - start) * blend
    return finite_edge, half_sine


def rapp_pa(signal, gain, saturation, smoothness):
    """Memoryless Rapp-style AM/AM response from rf_amplification_gain.py."""
    envelope = np.abs(signal)
    output_envelope = (gain * envelope /
                       (1.0 + (gain * envelope / saturation) ** (2.0 * smoothness))
                       ** (1.0 / (2.0 * smoothness)))
    return np.divide(signal * output_envelope, envelope,
                     out=np.zeros_like(signal), where=envelope > 0)


def welch_psd(signal, sample_rate, segment_samples=8192, overlap=0.5):
    """Two-sided Hann-windowed Welch PSD, implemented with NumPy."""
    segment_samples = min(segment_samples, len(signal))
    hop = max(1, int(segment_samples * (1.0 - overlap)))
    n_fft = 4 * segment_samples
    window = np.hanning(segment_samples)
    scale = sample_rate * np.sum(window ** 2)
    power = np.zeros(n_fft, dtype=float)
    count = 0
    for start in range(0, len(signal) - segment_samples + 1, hop):
        block = signal[start:start + segment_samples] * window
        spectrum = np.fft.fft(block, n=n_fft)
        power += np.abs(spectrum) ** 2 / scale
        count += 1
    power /= max(count, 1)
    frequency = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    return np.fft.fftshift(frequency), np.fft.fftshift(power)


def open_image(path):
    image_path = str(path.resolve())
    if sys.platform == 'darwin':
        subprocess.Popen(['open', image_path])
    elif os.name == 'nt':
        os.startfile(image_path)
    else:
        subprocess.Popen(['xdg-open', image_path])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits', type=int, default=4096,
                        help='input bits (default: 4096; must be even)')
    parser.add_argument('--seed', type=int, default=7, help='random stream seed (default: 7)')
    parser.add_argument('--symbol-rate', type=float, default=250_000,
                        help='OQPSK symbol rate in symbols/s (default: 250000)')
    parser.add_argument('--samples-per-bit', type=int, default=32,
                        help='baseband samples per serial bit (default: 32)')
    parser.add_argument('--edge-fraction', type=float, default=0.25,
                        help='finite branch transition duration as a fraction of a bit interval '
                             '(default: 0.25; illustrative, not an EFR32 specification)')
    parser.add_argument('--carrier-ghz', type=float, default=2.45,
                        help='frequency-axis carrier reference (default: 2.45 GHz)')
    parser.add_argument('--gain', type=float, default=3.0,
                        help='PA small-signal voltage gain (default: 3)')
    parser.add_argument('--saturation', type=float, default=2.4,
                        help='normalized PA saturation output (default: 2.4)')
    parser.add_argument('--smoothness', type=float, default=2.0,
                        help='Rapp model smoothness parameter (default: 2)')
    parser.add_argument('--span-mhz', type=float, default=3.0,
                        help='half-width of displayed spectrum (default: 3 MHz)')
    parser.add_argument('--output', default='oqpsk_shaping_pa_spectrum.png',
                        help='output PNG path')
    parser.add_argument('--no-open', action='store_true',
                        help='save the PNG without opening it')
    args = parser.parse_args()
    if args.bits <= 0 or args.bits % 2:
        parser.error('--bits must be a positive even number')
    if args.samples_per_bit < 4:
        parser.error('--samples-per-bit must be at least 4')
    if not 0 < args.edge_fraction <= 1:
        parser.error('--edge-fraction must be greater than 0 and at most 1')
    if min(args.symbol_rate, args.carrier_ghz, args.gain,
           args.saturation, args.smoothness, args.span_mhz) <= 0:
        parser.error('symbol rate, carrier, gain, saturation, smoothness, and span must be positive')

    rng = np.random.default_rng(args.seed)
    bits = rng.integers(0, 2, size=args.bits, dtype=np.int8)
    # Each input bit updates one branch; Q updates are staggered by half a
    # symbol. Both alternatives use the same bit sequence and timing.
    finite_edge_path, half_sine_path = make_oqpsk_pair(
        bits, args.samples_per_bit, args.edge_fraction)
    half_sine_pa = rapp_pa(half_sine_path, args.gain, args.saturation, args.smoothness)
    finite_edge_pa = rapp_pa(finite_edge_path, args.gain, args.saturation, args.smoothness)

    symbol_rate = args.symbol_rate
    bit_rate = 2.0 * symbol_rate
    sample_rate = bit_rate * args.samples_per_bit
    spectra = []
    for label, signal, color in [
        ('Half-sine pulse shaping, after PA', half_sine_pa, '#2878b5'),
        (f'Finite straight-path transitions ({args.edge_fraction:g} bit), after PA',
         finite_edge_pa, '#d14b40'),
    ]:
        offsets_hz, power = welch_psd(signal, sample_rate)
        spectra.append((label, color, offsets_hz, power))
    common_peak = max(float(power.max()) for _, _, _, power in spectra)

    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for label, color, offsets_hz, power in spectra:
        power_db = 10.0 * np.log10(np.maximum(power / common_peak, 1e-14))
        frequency_ghz = args.carrier_ghz + offsets_hz / 1e9
        ax.plot(frequency_ghz, power_db, color=color, lw=1.8, label=label)

    half_span_ghz = args.span_mhz / 1000.0
    ax.set_xlim(args.carrier_ghz - half_span_ghz,
                args.carrier_ghz + half_span_ghz)
    ax.set_ylim(-90, 3)
    ax.axvline(args.carrier_ghz, color='0.35', ls=':', lw=1.1,
               label=f'{args.carrier_ghz:g} GHz carrier')
    ax.set_xlabel(f'RF frequency (GHz), centered on {args.carrier_ghz:g} GHz')
    ax.set_ylabel('Relative power spectral density (dB; common peak reference)')
    ax.set_title(f'OQPSK spectrum: finite branch transitions vs half-sine shaping\n'
                 f'Same {args.bits:,}-bit stream at {symbol_rate / 1e3:g} ksymbols/s; both after PA')
    ax.grid(True, alpha=.24)
    ax.legend(loc='upper right', framealpha=.96)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output.resolve()} ({args.bits:,} bits, {args.bits // 2:,} OQPSK symbols, '
          f'{sample_rate / 1e6:g} MS/s; no DSSS)')
    if not args.no_open:
        open_image(output)


if __name__ == '__main__':
    main()
