"""
Generate comparison plots between Oobleck (singlenode) and standard DDP.

Produces a single figure with three panels:
  1. Loss curve over wall time
  2. Throughput (tokens/s) over wall time
  3. Fault recovery timeline (Gantt-style)

Usage:
    python plot_comparison.py \
        [--singlenode_csv singlenode_metrics.csv] \
        [--ddp_csv ddp_metrics.csv] \
        [--output comparison.png]

Both CSV files are produced automatically by run_gpt2.py and run_ddp_gpt2.py.
"""

import argparse
import csv
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, list]:
    """Return a dict of column-name -> list-of-values from a metrics CSV."""
    cols: dict[str, list] = {
        "wall_time": [], "step": [], "epoch": [],
        "loss": [], "tokens_per_sec": [], "event": [],
    }
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cols["wall_time"].append(float(row["wall_time"]))
                cols["step"].append(int(row["step"]) if row["step"] else -1)
                cols["epoch"].append(int(row["epoch"]) if row["epoch"] else -1)
                cols["loss"].append(float(row["loss"]) if row["loss"] else float("nan"))
                cols["tokens_per_sec"].append(
                    float(row["tokens_per_sec"]) if row["tokens_per_sec"] else float("nan")
                )
                cols["event"].append(row["event"])
    except FileNotFoundError:
        print(f"[warn] {path} not found — skipping.", file=sys.stderr)
    return cols


def _mask(cols: dict, event: str) -> np.ndarray:
    """Boolean mask for rows where event == event."""
    return np.array([e == event for e in cols["event"]])


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        valid = arr[lo : i + 1]
        valid = valid[~np.isnan(valid)]
        if len(valid):
            out[i] = valid.mean()
    return out


# ── Plot helpers ──────────────────────────────────────────────────────────────

OOBLECK_COLOR = "steelblue"
DDP_COLOR = "coral"
FAULT_COLOR = "darkorange"
CRASH_COLOR = "crimson"
RECOVER_COLOR = "mediumseagreen"


def _draw_event_vlines(ax, times, color, linestyle, label, drawn_labels):
    for t in times:
        lbl = label if label not in drawn_labels else "_nolegend_"
        ax.axvline(x=t, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.85, label=lbl)
        drawn_labels.add(label)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--singlenode_csv", default="singlenode_metrics.csv",
                        help="Metrics CSV from run_gpt2.py (default: singlenode_metrics.csv)")
    parser.add_argument("--ddp_csv", default="ddp_metrics.csv",
                        help="Metrics CSV from run_ddp_gpt2.py (default: ddp_metrics.csv)")
    parser.add_argument("--output", default="comparison.png",
                        help="Output image path (default: comparison.png)")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Rolling-average window for throughput plot (default: 20)")
    args = parser.parse_args()

    oo = load_csv(args.singlenode_csv)
    dp = load_csv(args.ddp_csv)

    if not oo["event"] and not dp["event"]:
        sys.exit("Both CSV files are missing or empty — nothing to plot.")

    # Convert to numpy for easier masking
    def _np(cols, key):
        return np.array(cols[key], dtype=float if key != "event" else object)

    oo_t   = _np(oo, "wall_time");   dp_t   = _np(dp, "wall_time")
    oo_ev  = np.array(oo["event"]);  dp_ev  = np.array(dp["event"])
    oo_loss = _np(oo, "loss");        dp_loss = _np(dp, "loss")
    oo_tps  = _np(oo, "tokens_per_sec"); dp_tps = _np(dp, "tokens_per_sec")

    oo_step_m  = oo_ev == "step";    dp_step_m  = dp_ev == "step"
    oo_fault_t = oo_t[oo_ev == "fault_detected"]
    oo_recov_t = oo_t[oo_ev == "recovered"]
    dp_crash_t = dp_t[dp_ev == "crash"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 16))
    fig.suptitle("Oobleck vs DDP — Fault Tolerance Comparison",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.38)

    # ── 1. Loss curve ──────────────────────────────────────────────────────
    ax1 = axes[0]
    drawn_labels: set = set()

    if oo_step_m.any():
        ax1.plot(oo_t[oo_step_m], oo_loss[oo_step_m],
                 color=OOBLECK_COLOR, linewidth=1.6, label="Oobleck")
    if dp_step_m.any():
        ax1.plot(dp_t[dp_step_m], dp_loss[dp_step_m],
                 color=DDP_COLOR, linewidth=1.6, label="DDP")

    _draw_event_vlines(ax1, oo_fault_t, FAULT_COLOR, "--", "Oobleck fault", drawn_labels)
    _draw_event_vlines(ax1, oo_recov_t, RECOVER_COLOR, ":", "Oobleck recovered", drawn_labels)
    _draw_event_vlines(ax1, dp_crash_t, CRASH_COLOR, "--", "DDP crash", drawn_labels)

    ax1.set_xlabel("Wall Time (s)"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # ── 2. Throughput ──────────────────────────────────────────────────────
    ax2 = axes[1]
    drawn_labels2: set = set()
    w = max(1, args.smooth)

    if oo_step_m.any():
        oo_tps_smooth = _rolling_mean(oo_tps[oo_step_m], w)
        ax2.plot(oo_t[oo_step_m], oo_tps_smooth,
                 color=OOBLECK_COLOR, linewidth=1.6, label="Oobleck")
    if dp_step_m.any():
        dp_tps_smooth = _rolling_mean(dp_tps[dp_step_m], w)
        ax2.plot(dp_t[dp_step_m], dp_tps_smooth,
                 color=DDP_COLOR, linewidth=1.6, label="DDP")

    _draw_event_vlines(ax2, oo_fault_t, FAULT_COLOR, "--", "Oobleck fault", drawn_labels2)
    _draw_event_vlines(ax2, oo_recov_t, RECOVER_COLOR, ":", "Oobleck recovered", drawn_labels2)
    _draw_event_vlines(ax2, dp_crash_t, CRASH_COLOR, "--", "DDP crash", drawn_labels2)

    ax2.set_xlabel("Wall Time (s)"); ax2.set_ylabel("Tokens / Second")
    ax2.set_title(f"Training Throughput Over Time  (rolling avg {w} steps)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── 3. Fault recovery timeline (Gantt) ─────────────────────────────────
    ax3 = axes[2]

    oo_max = float(oo_t.max()) if len(oo_t) else 0.0
    dp_max = float(dp_t.max()) if len(dp_t) else 0.0
    x_max  = max(oo_max, dp_max, 1.0)

    Y_OO, Y_DP = 1, 0   # y-band indices
    HEIGHT = 0.5

    def _gantt(ax, y, segments, color, label):
        """Draw one horizontal Gantt bar; segments is list of (start, width)."""
        first = True
        for start, width in segments:
            ax.broken_barh(
                [(start, width)], (y, HEIGHT),
                facecolors=color, edgecolors="white", linewidth=0.4,
                label=label if first else "_nolegend_",
            )
            first = False

    # Oobleck segments
    if len(oo_fault_t) > 0 and len(oo_recov_t) > 0:
        ft, rt = oo_fault_t[0], oo_recov_t[0]
        _gantt(ax3, Y_OO, [(0, ft)], OOBLECK_COLOR, "Training")
        _gantt(ax3, Y_OO, [(ft, rt - ft)], FAULT_COLOR, "Reconfiguring")
        if rt < oo_max:
            _gantt(ax3, Y_OO, [(rt, oo_max - rt)], OOBLECK_COLOR, "_nolegend_")
        downtime = rt - ft
        ax3.text((ft + rt) / 2, Y_OO + HEIGHT + 0.04,
                 f"↕ {downtime:.1f}s downtime",
                 ha="center", va="bottom", fontsize=9, color=FAULT_COLOR, fontweight="bold")
        ax3.annotate("", xy=(rt, Y_OO + HEIGHT * 0.5),
                     xytext=(ft, Y_OO + HEIGHT * 0.5),
                     arrowprops=dict(arrowstyle="<->", color=FAULT_COLOR, lw=1.5))
    else:
        _gantt(ax3, Y_OO, [(0, oo_max)], OOBLECK_COLOR, "Training")

    # DDP segments
    if len(dp_crash_t) > 0:
        ct = dp_crash_t[0]
        _gantt(ax3, Y_DP, [(0, ct)], DDP_COLOR, "Training")
        _gantt(ax3, Y_DP, [(ct, x_max - ct)], CRASH_COLOR, "Crashed (no recovery)")
        ax3.text(ct + (x_max - ct) / 2, Y_DP + HEIGHT + 0.04,
                 "No recovery", ha="center", va="bottom",
                 fontsize=9, color=CRASH_COLOR, fontweight="bold")
    else:
        _gantt(ax3, Y_DP, [(0, dp_max)], DDP_COLOR, "Training")

    ax3.set_yticks([Y_DP + HEIGHT / 2, Y_OO + HEIGHT / 2])
    ax3.set_yticklabels(["DDP", "Oobleck"], fontsize=11)
    ax3.set_ylim(-0.15, Y_OO + HEIGHT + 0.35)
    ax3.set_xlim(0, x_max * 1.02)
    ax3.set_xlabel("Wall Time (s)")
    ax3.set_title("Fault Recovery Timeline")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.2, axis="x")
    ax3.spines[["left", "right", "top"]].set_visible(False)
    ax3.tick_params(left=False)

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
