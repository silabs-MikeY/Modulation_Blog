#!/usr/bin/env python3
"""Plot an ideal 2.4 GHz IEEE 802.15.4 O-QPSK PHY frame constellation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import argparse
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mpl-dsss-constellation-cache'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE 802.15.4 2.4 GHz symbol (0..15) -> 32 chips, c0 transmitted first.
CHIP_MAP = [
    '11011001110000110101001000101110', '11101101100111000011010100100010',
    '00101110110110011100001101010010', '00100010111011011001110000110101',
    '01010010001011101101100111000011', '00110101001000101110110110011100',
    '11000011010100100010111011011001', '10011100001101010010001011101101',
    '10001100100101100000011101111011', '10111000110010010110000001110111',
    '01111011100011001001011000000111', '01110111101110001100100101100000',
    '00000111011110111000110010010110', '01100000011101111011100011001001',
    '10010110000001110111101110001100', '11001001011000000111011110111000',
]

# A valid-format example PSDU payload; not a captured over-the-air packet.
psdu = bytes.fromhex('41 88 01 00 00 00 00 00')
phy_bytes = bytes([0, 0, 0, 0, 0xA7, len(psdu)]) + psdu  # preamble, SFD, PHR, PSDU
bits = [(byte >> bit) & 1 for byte in phy_bytes for bit in range(8)]  # LSB first
symbols = [sum(bits[k+n] << n for n in range(4)) for k in range(0, len(bits), 4)]
chips = ''.join(CHIP_MAP[s] for s in symbols)
chip_signs = np.array([1.0 if b == '1' else -1.0 for b in chips])

# Half-sine pulse of duration 2Tc; Q chips are offset from I by Tc.
samples_per_chip = 24
t = np.arange(0, len(chip_signs), 1 / samples_per_chip)
i = np.zeros_like(t); q = np.zeros_like(t)
for k, c in enumerate(chip_signs):
    p = np.sin(np.pi * (t-k) / 2)
    p[(t-k < 0) | (t-k > 2)] = 0
    if k % 2 == 0: i += c*p
    else: q += c*p

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(i, q, c=t / 32, s=7, cmap='viridis', alpha=.75, linewidths=0,
                label='Transmitted complex-envelope samples')
ax.plot(np.cos(np.linspace(0, 2*np.pi, 500)), np.sin(np.linspace(0, 2*np.pi, 500)),
        color='0.45', linestyle=':', linewidth=1.2, label='Unit-envelope reference')
# Mark states at chip boundaries after the initial startup ramp.
bt = np.arange(1, len(chip_signs)+1, 1.0)
bi = np.interp(bt, t, i); bq = np.interp(bt, t, q)
ax.scatter(bi, bq, s=12, facecolors='none', edgecolors='black', linewidths=.5,
           label='Chip-boundary samples')
ax.axhline(0, color='0.65', linewidth=.8); ax.axvline(0, color='0.65', linewidth=.8)
ax.set(xlim=(-1.25,1.25), ylim=(-1.25,1.25), xlabel='In-phase component I (normalized)',
       ylabel='Quadrature component Q (normalized)',
       title='2.4 GHz IEEE 802.15.4 O-QPSK: example PHY frame constellation')
ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.18)
cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.04); cb.set_label('4-bit PHY symbols from frame start')
ax.legend(loc='lower left', fontsize=8, framealpha=.95)
fig.text(.5, .015, 'Ideal simulated PHY waveform • 4-byte preamble + SFD + PHR + 8-byte PSDU • 32 chips per 4-bit symbol',
         ha='center', color='0.35', fontsize=9)
fig.tight_layout(rect=(0, .04, 1, 1))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--no-open', action='store_true', help='save the PNG without opening it')
parser.add_argument('--output', default='dsss_802154_actual_constellation.png', help='output PNG path')
args = parser.parse_args()
output = Path(args.output).resolve()
fig.savefig(output, dpi=180)
plt.close(fig)
print(f'Saved {output} ({len(phy_bytes)} PHY bytes, {len(symbols)} symbols, {len(chips)} chips)')

if not args.no_open:
    if sys.platform == 'darwin':
        subprocess.Popen(['open', str(output)])
    elif os.name == 'nt':
        os.startfile(output)
    else:
        subprocess.Popen(['xdg-open', str(output)])
