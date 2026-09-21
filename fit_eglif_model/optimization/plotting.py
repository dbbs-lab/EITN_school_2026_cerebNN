import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .features import multicomp_features
from .fitness import _fi_curve


def parse_current_from_filename(fname: str) -> float:
    m = re.search(r"inj(-?\d+(?:\.\d+)?)", fname)
    return float(m.group(1)) if m else np.nan


def flatten_peak_times(peak_values):
    out = []
    for x in peak_values:
        if x is None:
            continue
        if isinstance(x, (list, tuple, np.ndarray)):
            out.extend(np.asarray(x, dtype=float).ravel().tolist())
        else:
            try:
                out.append(float(x))
            except Exception:
                continue
    out = np.asarray(out, dtype=float)
    return out[np.isfinite(out)]


def style_vm_axis(ax):
    ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=1)
    ax.tick_params(axis="both", which="minor", labelsize=10, length=2, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", labelbottom=False)


def style_I_axis(axI, show_xlabel=False):
    axI.spines["top"].set_visible(False)
    axI.spines["left"].set_visible(False)
    axI.spines["right"].set_visible(True)
    axI.spines["bottom"].set_visible(True)

    axI.spines["right"].set_linewidth(1)
    axI.spines["right"].set_color("#7a1f2b")
    axI.spines["bottom"].set_linewidth(1)

    axI.yaxis.set_ticks_position("right")
    axI.yaxis.set_label_position("right")
    axI.tick_params(
        axis="y", which="major", labelsize=5, colors="#7a1f2b", length=3, width=1, pad=2
    )
    axI.set_ylabel("I [pA]", color="#7a1f2b", fontsize=6, labelpad=3)
    axI.tick_params(axis="x", which="major", labelsize=9, length=3, width=1, pad=12)
    axI.tick_params(axis="x", which="minor", length=2, width=0.8, pad=12)

    if show_xlabel:
        axI.set_xlabel("Time [ms]", fontsize=10, labelpad=8)
    else:
        axI.set_xlabel("")
    for lab in axI.get_xticklabels(which="both"):
            lab.set_clip_on(False)


def plot_experiment_traces(traces, protocol, currents):
    major_locator = mticker.MaxNLocator(nbins=5)
    minor_locator = mticker.AutoMinorLocator(2)

    n = len(traces)
    fig = plt.figure(figsize=(7.6, 2.65 * n), constrained_layout=False)

    # 3 suplots per trace
    height_ratios = np.tile([4.0, 1.6, 0.55], n)
    # last suplot get more space to write legend
    height_ratios[-1] = 0.15
    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=1,
        height_ratios=height_ratios,
        hspace=0.10,
    )

    time_bounds = np.asarray([[t["T"][0], t["T"][-1]] for t in traces])
    global_tmin = np.min(time_bounds[:, 0])
    global_tmax = np.max(time_bounds[:, 1])
    Imax = max(1.0, max(abs(t) for t in currents))
    Iylim = 1.10 * Imax
    Iticks = [-Imax, 0.0, Imax]

    for i, trace in enumerate(traces):
        row = i * 3
                
        # highlight stimulus time except when no stim is applied
        axV = fig.add_subplot(gs[row, 0])
        axV.set_facecolor("white")
        if currents[i] != 0:
            axV.axvspan(
                trace["stim_start"][0],
                trace["stim_end"][0],
                facecolor="#ffe680",
                alpha=0.18,
                edgecolor="none",
                zorder=0,
            )
        
        # Vm trace + threshold
        axV.plot(
            trace["T"], trace["V"], 
            color="0.35", linewidth=1.1, zorder=2
        )
        axV.axhline(
            y=protocol['threshold'], 
            color="0.6", linestyle="--", linewidth=0.9, zorder=1
        )

        axV.set_ylabel(r"$V_m$ [mV]", fontsize=11)
        axV.set_ylim(-100, 25)
        axV.set_xlim(global_tmin, global_tmax)
        style_vm_axis(axV)

        # Input current trace (0 outside stim)
        axI = fig.add_subplot(gs[row + 1, 0], sharex=axV)
        axI.set_facecolor("white")
        I = np.zeros_like(trace["T"], dtype=float)
        I[(trace["T"] >= protocol["stim_start"]) * 
          (trace["T"] <= protocol["stim_end"])] = currents[i]
        axI.plot(
            trace["T"], I, 
            drawstyle="steps-post", linewidth=1.6, color="#7a1f2b"
        )
        axI.set_ylim(-Iylim, Iylim)
        axI.set_yticks(Iticks)
        style_I_axis(axI, show_xlabel=i==n - 1)

        txt = axI.text(
            0.98,
            0.30 if currents[i] >= 0 else 0.70,
            f"I = {currents[i]:g} pA",
            transform=axI.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            color="#7a1f2b",
            bbox=dict(
                boxstyle="round,pad=0.15", 
                facecolor="white", 
                edgecolor="none", 
                alpha=0.80
            ),
            zorder=10,
        )
        txt.set_clip_on(True)
        txt.set_clip_path(axI.patch)
    fig.subplots_adjust(top=0.985, bottom=0.085, left=0.12, right=0.93)
    return fig


def plot_fi_curve(neuron_features, nest_features, labels=["target", "nest model"]):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.set_facecolor("white")
    I_target, f_target = _fi_curve(neuron_features)
    ax.plot(I_target, f_target, '.-k', label=labels[0])
    I_fit, f_fit = _fi_curve(nest_features)
    ax.plot(I_fit, f_fit, '.-r', label=labels[1])
    ax.set_xlabel('inj current [pA]')
    ax.set_ylabel('spks/s')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_raster(neuron_features, nest_features, inj_currents, protocol, labels=['target', 'nest model']):
    
    fig, ax = plt.subplots(figsize=(5,3))
    ax.set_facecolor("white")
    for i, I in enumerate(inj_currents):
        if np.all(neuron_features.loc[neuron_features['current'] == I, 'peak_time'].values[0]):
            spks_neuron = neuron_features.loc[neuron_features["current"] == I, "peak_time"].values[0]
            plt.plot(spks_neuron, np.repeat(I, len(spks_neuron)), '|k', markersize=4)
        if len(nest_features.loc[nest_features['current'] == I, 'peak_time'].values[0]) > 0:
            spks_nest = nest_features.loc[nest_features["current"] == I, "peak_time"].values[0]
            plt.plot(spks_nest, np.repeat(I, len(spks_nest)), '|r', markersize=4)
    plt.xlabel('time [ms]')
    plt.ylabel('inj current [pA]')
    plt.xlim([0, protocol['duration']])
    plt.ylim([np.min(inj_currents) - 1, np.max(inj_currents) + 1])
    plt.plot([protocol['stim_start'], protocol['stim_start']], [np.min(inj_currents), np.max(inj_currents)], '-b')
    plt.plot([protocol['stim_end'], protocol['stim_end']], [np.min(inj_currents), np.max(inj_currents)], '-b')
    plt.tight_layout()
    return fig