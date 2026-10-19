#!/usr/bin/env python3
"""Plot the half-sine-shaped trajectory with square O-QPSK reference states."""
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

CHIPS_0000 = '11011001110000110101001000101110'  # c0 first

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dsss_constellation_half_sine.png', help='output PNG path')
    parser.add_argument('--no-open', action='store_true', help='save the PNG without opening it')
    args = parser.parse_args()

    chips = np.array([1.0 if bit == '1' else -1.0 for bit in CHIPS_0000])
    t = np.linspace(1.0, len(chips), (len(chips) - 1) * 500 + 1)
    i = np.zeros_like(t)
    q = np.zeros_like(t)
    for k, chip in enumerate(chips):
        tau = t - k
        pulse = np.where((tau >= 0) & (tau <= 2), np.sin(np.pi * tau / 2), 0.0)
        if k % 2 == 0:
            i += chip * pulse
        else:
            q += chip * pulse

    fig, ax = plt.subplots(figsize=(8, 8))
    points = np.column_stack((i, q))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    midpoints = (points[:-1] + points[1:]) / 2
    phase = np.mod(np.rad2deg(np.arctan2(midpoints[:, 1], midpoints[:, 0])), 360)
    # On the upper/lower arcs, the horizontal-branch bit changes; on the
    # left/right arcs, the delayed vertical-branch bit changes.
    bit1_arc = (((phase >= 45) & (phase <= 135)) |
                ((phase >= 225) & (phase <= 315)))
    segment_colors = np.where(bit1_arc, '#2878b5', '#e07a24')
    trajectory = LineCollection(segments, colors=segment_colors, linewidths=2.7,
                                capstyle='round', joinstyle='round', zorder=2)
    ax.add_collection(trajectory)
    angles = np.deg2rad([45, 135, 225, 315])
    states = np.column_stack((np.cos(angles), np.sin(angles)))
    ax.scatter(states[:, 0], states[:, 1], s=78, color='#d62728',
               edgecolors='black', linewidths=1.2, zorder=4)
    labels = [('01', '45°'), ('00', '135°'), ('10', '225°'), ('11', '315°')]
    label_offsets = [(12, 10), (-12, 10), (12, -45), (-12, -45)]
    label_alignments = [('left', 'bottom'), ('right', 'bottom'),
                        ('left', 'top'), ('right', 'top')]
    for (bits, degrees), (x, y), offset, align in zip(labels, states,
                                                       label_offsets, label_alignments):
        ax.annotate(f'{bits}\n{degrees}', (x, y), xytext=offset,
                    textcoords='offset points', ha=align[0], va=align[1],
                    fontsize=10, fontweight='bold', linespacing=1.15,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=1.5),
                    zorder=5)
    # Large double-headed arrows follow each shaped quarter-circle arc.
    def add_arc_arrow(start_degrees, end_degrees, color):
        direction = np.sign(end_degrees - start_degrees)
        # Leave an obvious visual gap between each arrow tip and the state marker.
        gap_degrees = 8
        start = np.deg2rad(start_degrees + direction * gap_degrees)
        end = np.deg2rad(end_degrees - direction * gap_degrees)
        delta = end - start
        tangent_scale = 4 / 3 * np.tan(abs(delta) / 4) * np.sign(delta)
        p0 = np.array([np.cos(start), np.sin(start)])
        p3 = np.array([np.cos(end), np.sin(end)])
        tangent0 = np.array([-np.sin(start), np.cos(start)])
        tangent3 = np.array([-np.sin(end), np.cos(end)])
        p1 = p0 + tangent_scale * tangent0
        p2 = p3 - tangent_scale * tangent3
        arc_path = MplPath([p0, p1, p2, p3],
                           [MplPath.MOVETO, MplPath.CURVE4,
                            MplPath.CURVE4, MplPath.CURVE4])
        arrow = FancyArrowPatch(path=arc_path, arrowstyle='<->', mutation_scale=23,
                                linewidth=3.6, color=color, shrinkA=0, shrinkB=0,
                                zorder=4)
        ax.add_patch(arrow)

    for start_degrees, end_degrees, color in [
        (45, 135, '#2878b5'), (225, 315, '#2878b5'),
        (135, 225, '#e07a24'), (315, 405, '#e07a24'),
    ]:
        add_arc_arrow(start_degrees, end_degrees, color)
    ax.axhline(0, color='0.65', lw=.8)
    ax.axvline(0, color='0.65', lw=.8)
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), xlabel='Real component (normalized)',
           ylabel='Imaginary component (normalized)',
           title='Half-sine-shaped O-QPSK trajectory and square reference constellation')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)
    legend_items = [
        Line2D([0], [0], color='#2878b5', lw=2.7,
               label='bit1 branch transition (upper/lower arcs)'),
        Line2D([0], [0], color='#e07a24', lw=2.7,
               label='bit0 branch transition (left/right arcs, delayed)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#d62728',
               markeredgecolor='black', markersize=8, label='bit pair and phase state'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8.5, framealpha=.95)
    fig.tight_layout()
    output = Path(args.output).resolve()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f'Saved {output}')
    if not args.no_open:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', str(output)])
        elif os.name == 'nt':
            os.startfile(output)
        else:
            subprocess.Popen(['xdg-open', str(output)])
if __name__ == '__main__':
    main()
