"""Plot a four-cycle BPSK transition and its spectrum.

Run with: python psk_smooth_phase_transition.py

The transition uses a cosine-shaped amplitude trajectory from +1 to -1.
It passes through the origin, as a pulse-shaped antipodal signal can.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    carrier_frequency = 2.45e9
    cycles_before = 1
    transition_cycles = 4
    cycles_after = 1
    total_cycles = cycles_before + transition_cycles + cycles_after
    samples_per_cycle = 240

    # Use carrier cycles as the horizontal coordinate to keep the four-cycle
    # transition easy to see; the frequency is shown in the figure title.
    time_cycles = np.linspace(
        0, total_cycles, total_cycles * samples_per_cycle, endpoint=False
    )
    transition_start = float(cycles_before)
    transition_end = float(cycles_before + transition_cycles)
    u = np.clip((time_cycles - transition_start) / transition_cycles, 0, 1)

    # Cosine-shaped interpolation of the carrier multiplier, +1 to -1.
    envelope = np.ones_like(time_cycles)
    in_transition = (time_cycles >= transition_start) & (time_cycles <= transition_end)
    envelope[in_transition] = np.cos(np.pi * u[in_transition])
    envelope[time_cycles > transition_end] = -1
    waveform = envelope * np.sin(2 * np.pi * time_cycles)

    snapshot_progress = np.linspace(0, 1, 5)
    snapshot_cycles = transition_start + snapshot_progress * transition_cycles
    snapshot_envelopes = np.cos(np.pi * snapshot_progress)

    fig = plt.figure(figsize=(13, 7.5), num="BPSK transition through origin")
    grid = fig.add_gridspec(2, 5, height_ratios=[2.2, 1.35], hspace=0.48, wspace=0.35)
    waveform_ax = fig.add_subplot(grid[0, :])
    snapshot_axes = [fig.add_subplot(grid[1, column]) for column in range(5)]
    fig.suptitle(
        "BPSK symbol transition through the origin\n"
        "Cosine-shaped amplitude transition over 4 carrier cycles "
        f"({carrier_frequency / 1e9:.2f} GHz carrier)",
        fontsize=14,
    )

    waveform_ax.plot(time_cycles, waveform, color="tab:blue", linewidth=1.6)
    waveform_ax.axhline(0, color="0.65", linewidth=0.8)
    waveform_ax.axvspan(transition_start, transition_end,
                        color="tab:orange", alpha=0.12, label="4-cycle transition")
    for index, cycle in enumerate(snapshot_cycles):
        waveform_ax.axvline(cycle, color="0.45", linestyle=":", linewidth=1)
        waveform_ax.plot(cycle, np.interp(cycle, time_cycles, waveform),
                         marker="o", color="tab:red", markersize=5, zorder=3)
        waveform_ax.text(cycle, 1.18, str(index + 1), ha="center", va="bottom",
                         fontsize=9, color="tab:red")
    waveform_ax.set_xlim(0, total_cycles)
    waveform_ax.set_ylim(-1.3, 1.4)
    waveform_ax.set_xticks(np.arange(total_cycles + 1))
    waveform_ax.set_xlabel("Time (carrier cycles)")
    waveform_ax.set_ylabel("Signal amplitude")
    waveform_ax.set_title("Carrier waveform and constellation snapshot locations")
    waveform_ax.grid(True, alpha=0.25)
    waveform_ax.legend(loc="lower left")

    for index, (snapshot_ax, progress, signed_amplitude) in enumerate(
        zip(snapshot_axes, snapshot_progress, snapshot_envelopes)
    ):
        snapshot_ax.axhline(0, color="0.78", linewidth=1)
        snapshot_ax.axvline(0, color="0.78", linewidth=1)
        # The two ideal BPSK symbol locations remain as gray references.
        snapshot_ax.scatter([1, -1], [0, 0], color="0.78", s=34, zorder=2)
        snapshot_ax.scatter([signed_amplitude], [0], color="tab:red",
                            edgecolor="black", s=85, zorder=3)
        snapshot_ax.set_xlim(-1.3, 1.3)
        snapshot_ax.set_ylim(-0.8, 0.8)
        snapshot_ax.set_aspect("equal", adjustable="box")
        snapshot_ax.set_xticks([-1, 0, 1])
        snapshot_ax.set_yticks([])
        snapshot_ax.grid(True, axis="x", alpha=0.2)
        snapshot_ax.set_title(f"{progress * 100:.0f}%\n{index + 1} cycle")
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))

    # Spectrum of the finite waveform shown above.
    fft_size = 65536
    sample_rate = carrier_frequency * samples_per_cycle
    frequencies_ghz = np.fft.rfftfreq(fft_size, d=1 / sample_rate) / 1e9

    def magnitude_db(signal):
        magnitude = 2 * np.abs(np.fft.rfft(signal, n=fft_size)) / len(signal)
        return 20 * np.log10(np.maximum(magnitude, 1e-4))

    spectrum_fig, spectrum_ax = plt.subplots(
        figsize=(10, 5), num="Gradual BPSK transition spectrum"
    )
    spectrum_ax.plot(frequencies_ghz, magnitude_db(waveform),
                     color="tab:blue", linewidth=1.4,
                     label="Gradual transition over 4 cycles")
    spectrum_ax.axvline(carrier_frequency / 1e9, color="0.35", linestyle="--",
                        linewidth=1, label="Carrier: 2.45 GHz")
    spectrum_ax.set_title("Spectrum of the gradual BPSK transition")
    spectrum_ax.set_xlabel("Frequency (GHz)")
    spectrum_ax.set_ylabel("Magnitude (dB relative to symbol amplitude)")
    spectrum_ax.set_xlim(0, 30)
    spectrum_ax.set_ylim(-80, 5)
    spectrum_ax.grid(True, alpha=0.25)
    spectrum_fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
