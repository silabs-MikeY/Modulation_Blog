"""Show the four QPSK states, waveforms, constellation, and bit mapping.

Run with: python psk_qpsk_states.py
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    amplitude = 1.0
    carrier_frequency = 2.45e9
    samples_per_cycle = 240
    time_cycles = np.linspace(0, 1, samples_per_cycle, endpoint=False)
    time_ns = time_cycles / carrier_frequency * 1e9

    # Square QPSK constellation with Gray-coded neighboring symbols. Other
    # systems may rotate or relabel it while preserving the 90-degree spacing.
    states = [
        {"bits": "00", "phase_deg": 45, "color": "tab:blue"},
        {"bits": "01", "phase_deg": 135, "color": "tab:orange"},
        {"bits": "11", "phase_deg": 225, "color": "tab:green"},
        {"bits": "10", "phase_deg": 315, "color": "tab:red"},
    ]
    constellation = np.array([
        [amplitude * np.cos(np.deg2rad(state["phase_deg"])),
         amplitude * np.sin(np.deg2rad(state["phase_deg"]))]
        for state in states
    ])

    fig, axes = plt.subplots(
        4, 2, figsize=(12, 12),
        gridspec_kw={"width_ratios": [2.1, 1]},
    )
    fig.suptitle(
        "QPSK: four symbols, four phases\n"
        "Square constellation, Gray mapping: 00 → 45°, 01 → 135°, "
        "11 → 225°, 10 → 315°",
        fontsize=14,
    )

    for row, state in enumerate(states):
        phase = np.deg2rad(state["phase_deg"])
        color = state["color"]

        wave_ax = axes[row, 0]
        waveform = amplitude * np.sin(2 * np.pi * time_cycles + phase)
        wave_ax.plot(time_ns, waveform, color=color, linewidth=1.8)
        wave_ax.axhline(amplitude, color="0.55", linestyle="--", linewidth=1)
        wave_ax.axhline(-amplitude, color="0.55", linestyle="--", linewidth=1)
        wave_ax.set_title(
            f"Bits {state['bits']}  |  phase {state['phase_deg']}°", loc="left"
        )
        wave_ax.set_ylabel("Signal amplitude")
        wave_ax.set_xlabel("Time (ns)")
        wave_ax.text(
            0.98, 0.96, f"Carrier: {carrier_frequency / 1e9:.2f} GHz",
            transform=wave_ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                  "pad": 2},
        )
        wave_ax.set_ylim(-1.25 * amplitude, 1.25 * amplitude)
        wave_ax.grid(True, alpha=0.25)

        constellation_ax = axes[row, 1]
        constellation_ax.axhline(0, color="0.75", linewidth=1)
        constellation_ax.axvline(0, color="0.75", linewidth=1)
        constellation_ax.scatter(constellation[:, 0], constellation[:, 1],
                                 color="0.78", s=65, zorder=2)
        active_x, active_y = constellation[row]
        constellation_ax.scatter([active_x], [active_y], color=color,
                                 edgecolor="black", s=130, zorder=3)

        for point, other_state in zip(constellation, states):
            x_offset = 6 if point[0] >= 0 else -20
            y_offset = 7 if point[1] >= 0 else -16
            constellation_ax.annotate(
                other_state["bits"], point, xytext=(x_offset, y_offset),
                textcoords="offset points", fontsize=9,
                color=other_state["color"] if other_state is state else "0.4",
            )

        if row == 0:
            constellation_ax.set_title("Constellation")
        else:
            constellation_ax.set_title("")
        constellation_ax.set_xlabel("cos" if row == len(states) - 1 else "")
        constellation_ax.set_ylabel("sin")
        constellation_ax.set_xlim(-1.4 * amplitude, 1.4 * amplitude)
        constellation_ax.set_ylim(-1.4 * amplitude, 1.4 * amplitude)
        constellation_ax.set_aspect("equal", adjustable="box")
        constellation_ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


if __name__ == "__main__":
    main()
