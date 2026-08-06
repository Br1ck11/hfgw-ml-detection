"""Pre-training plots that verify signal injection and preprocessing outputs."""

from __future__ import annotations

import csv
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np


OVERLAP_INCIDENT_COLUMNS = [
    "split",
    "source_file",
    "event_id_a",
    "event_id_b",
    "mass_solar_a",
    "mass_solar_b",
    "target_peak_snr_a",
    "target_peak_snr_b",
    "start_sample_a",
    "end_sample_a",
    "start_sample_b",
    "end_sample_b",
    "overlap_start_sample",
    "overlap_end_sample",
    "overlap_samples",
]


def _open_window_memmap(info, split, key):
    path = info.get(f"{split}_{key}_path")
    if not path or not os.path.exists(path):
        return None

    base_shape = tuple(info[f"{split}_shape"])
    if key == "lbl":
        return np.memmap(path, mode="r", dtype=np.bool_, shape=(base_shape[0],))
    return np.memmap(path, mode="r", dtype=np.dtype(info["dtype"]), shape=base_shape)


def _sample_candidates(indices, max_candidates):
    if indices.size <= max_candidates:
        return indices
    positions = np.linspace(0, indices.size - 1, max_candidates, dtype=np.int64)
    return indices[positions]


def _choose_representative_noise_index(noisy_windows, noise_indices, max_candidates):
    candidates = _sample_candidates(noise_indices, max_candidates)
    values = np.asarray(noisy_windows[candidates], dtype=np.float64)
    rms = np.sqrt(np.mean(values.reshape(values.shape[0], -1) ** 2, axis=1))
    return int(candidates[np.argmin(np.abs(rms - np.median(rms)))])


def _choose_strong_signal_index(clean_windows, signal_indices, max_candidates):
    candidates = _sample_candidates(signal_indices, max_candidates)
    if clean_windows is None:
        return int(candidates[0])
    values = np.asarray(clean_windows[candidates], dtype=np.float64)
    peak = np.max(np.abs(values.reshape(values.shape[0], -1)), axis=1)
    return int(candidates[np.argmax(peak)])


def _read_event_metadata(event_path):
    events = []
    with open(event_path, newline="") as handle:
        for row in csv.DictReader(handle):
            events.append(
                {
                    "event_id": int(row["event_id"]),
                    "split": row["split"],
                    "source_file": row["source_file"],
                    "start": int(row["injection_start_sample"]),
                    "end": int(row["injection_end_sample"]),
                    "mass_solar": float(row["mass_solar"]),
                    "target_peak_snr": float(row["target_peak_snr"]),
                    "snr_peak": float(row["snr_peak"]),
                    "noise_std": float(row["noise_std"]),
                    "response_mode": row.get("response_mode", ""),
                }
            )
    return events


def save_overlapping_injection_report(preprocessing_info, output_dir):
    """Save direct clean-waveform overlap incidences and aggregate counts."""
    event_path = preprocessing_info.get("event_metadata_path")
    if not event_path or not os.path.exists(event_path):
        warnings.warn(
            "Cannot create overlapping-injection report because event metadata "
            "is unavailable."
        )
        return {}

    events = _read_event_metadata(event_path)
    events_by_group = {}
    for event in events:
        events_by_group.setdefault(
            (event["split"], event["source_file"]), []
        ).append(event)

    incidents = []
    involved_event_ids = set()
    involved_by_split = {}
    incident_count_by_split = {}
    for (split, source_file), group_events in events_by_group.items():
        active = []
        for event in sorted(group_events, key=lambda item: (item["start"], item["end"])):
            active = [other for other in active if other["end"] > event["start"]]
            for other in active:
                overlap_start = max(other["start"], event["start"])
                overlap_end = min(other["end"], event["end"])
                if overlap_end <= overlap_start:
                    continue
                incidents.append(
                    {
                        "split": split,
                        "source_file": source_file,
                        "event_id_a": other["event_id"],
                        "event_id_b": event["event_id"],
                        "mass_solar_a": other["mass_solar"],
                        "mass_solar_b": event["mass_solar"],
                        "target_peak_snr_a": other["target_peak_snr"],
                        "target_peak_snr_b": event["target_peak_snr"],
                        "start_sample_a": other["start"],
                        "end_sample_a": other["end"],
                        "start_sample_b": event["start"],
                        "end_sample_b": event["end"],
                        "overlap_start_sample": overlap_start,
                        "overlap_end_sample": overlap_end,
                        "overlap_samples": overlap_end - overlap_start,
                    }
                )
                involved_event_ids.update((other["event_id"], event["event_id"]))
                involved_by_split.setdefault(split, set()).update(
                    (other["event_id"], event["event_id"])
                )
                incident_count_by_split[split] = incident_count_by_split.get(split, 0) + 1
            active.append(event)

    os.makedirs(output_dir, exist_ok=True)
    incidents_path = os.path.join(output_dir, "overlapping_injection_incidents.csv")
    with open(incidents_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OVERLAP_INCIDENT_COLUMNS)
        writer.writeheader()
        writer.writerows(incidents)

    event_count_by_split = {}
    for event in events:
        event_count_by_split[event["split"]] = (
            event_count_by_split.get(event["split"], 0) + 1
        )

    summary_rows = []
    for split in sorted(event_count_by_split):
        event_count = event_count_by_split[split]
        unique_involved = len(involved_by_split.get(split, set()))
        summary_rows.append(
            {
                "split": split,
                "num_injected_events": event_count,
                "num_direct_overlap_pair_incidents": incident_count_by_split.get(split, 0),
                "num_unique_events_involved_in_direct_overlap": unique_involved,
                "fraction_events_involved_in_direct_overlap": (
                    unique_involved / event_count if event_count else 0.0
                ),
            }
        )
    summary_rows.append(
        {
            "split": "all",
            "num_injected_events": len(events),
            "num_direct_overlap_pair_incidents": len(incidents),
            "num_unique_events_involved_in_direct_overlap": len(involved_event_ids),
            "fraction_events_involved_in_direct_overlap": (
                len(involved_event_ids) / len(events) if events else 0.0
            ),
        }
    )

    summary_path = os.path.join(output_dir, "overlapping_injection_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    all_row = summary_rows[-1]
    print(
        "[PRETRAIN CHECK] Direct injection overlaps: "
        f"{all_row['num_direct_overlap_pair_incidents']} pair incidences; "
        f"{all_row['num_unique_events_involved_in_direct_overlap']}/"
        f"{all_row['num_injected_events']} events involved."
    )
    return {
        "overlap_incidents": incidents_path,
        "overlap_summary": summary_path,
    }


def _isolated_fully_contained_candidates(info, split, signal_indices, max_candidates):
    """
    Return candidate indices containing exactly one fully-contained event.

    Selecting only by maximum clean amplitude preferentially finds rare
    overlapping injections, because |s1 + s2| can exceed either event alone.
    """
    event_path = info.get("event_metadata_path")
    window_path = info.get("window_metadata_path")
    if not event_path or not window_path:
        return np.empty(0, dtype=np.int64)
    if not os.path.exists(event_path) or not os.path.exists(window_path):
        return np.empty(0, dtype=np.int64)

    sampled = _sample_candidates(signal_indices, max_candidates)
    sampled_set = {int(index) for index in sampled}
    candidate_windows = {}
    with open(window_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            index = int(row["window_index"])
            if index not in sampled_set:
                continue
            candidate_windows[index] = {
                "event_id": int(row["event_id"]),
                "start": int(row["window_start_sample"]),
                "end": int(row["window_end_sample"]),
                "overlap_fraction": float(row["overlap_fraction"]),
            }
            if len(candidate_windows) == len(sampled_set):
                break

    events_by_group = {}
    events_by_id = {}
    for event in _read_event_metadata(event_path):
        events_by_id[event["event_id"]] = event
        events_by_group.setdefault(
            (event["split"], event["source_file"]), []
        ).append(event)

    isolated = []
    for index, window in candidate_windows.items():
        event = events_by_id.get(window["event_id"])
        if event is None or window["overlap_fraction"] < 1.0 - 1e-9:
            continue
        if event["start"] < window["start"] or event["end"] > window["end"]:
            continue

        group_events = events_by_group[(event["split"], event["source_file"])]
        overlap_count = sum(
            other["start"] < window["end"] and other["end"] > window["start"]
            for other in group_events
        )
        if overlap_count == 1:
            isolated.append(index)
    return np.asarray(isolated, dtype=np.int64)


def _selected_event_metadata(info, split, signal_index):
    event_path = info.get("event_metadata_path")
    window_path = info.get("window_metadata_path")
    if not event_path or not window_path:
        return None
    if not os.path.exists(event_path) or not os.path.exists(window_path):
        return None

    event_id = None
    with open(window_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == split and int(row["window_index"]) == signal_index:
                event_id = int(row["event_id"])
                break
    if event_id is None or event_id < 0:
        return None

    for event in _read_event_metadata(event_path):
        if event["event_id"] == event_id:
            return event
    return None


def _event_title_suffix(event):
    if event is None:
        return "event metadata unavailable"
    return (
        f"peak SNR={event['target_peak_snr']:g} | "
        f"mass={event['mass_solar']:.3e} M_solar | "
        f"response={event['response_mode']}"
    )


def _window_magnitude_peak(window):
    values = _as_channels(window)
    if values.shape[1] == 1:
        return float(np.max(np.abs(values[:, 0])))
    return float(np.max(np.sqrt(np.sum(values[:, :2] ** 2, axis=1))))


def _validate_selected_event_alignment(clean_raw_window, event, signal_selection):
    if event is None or "isolated fully-contained" not in signal_selection:
        return
    noise_std = float(event["noise_std"])
    if not np.isfinite(noise_std) or noise_std <= 0:
        return

    measured_peak_snr = _window_magnitude_peak(clean_raw_window) / noise_std
    expected_peak_snr = float(event["snr_peak"])
    if not np.isclose(measured_peak_snr, expected_peak_snr, rtol=5e-3, atol=5e-3):
        raise RuntimeError(
            "Selected clean-signal window does not match its event metadata: "
            f"clean peak/noise_std={measured_peak_snr:.6g}, but metadata "
            f"snr_peak={expected_peak_snr:.6g}. The shared memmap directory "
            "was likely overwritten by another preprocessing run. Refusing to "
            "create a falsely labeled training-comparison plot."
        )


def _as_channels(window):
    values = np.asarray(window)
    if values.ndim == 1:
        return values[:, None]
    return values


def _time_axis(info, length):
    fs = info.get("fs_val")
    if fs is not None and np.isfinite(fs) and float(fs) > 0:
        return np.arange(length) / float(fs) * 1e6, "time (us)"
    return np.arange(length), "sample"


def _plot_window(window, title, ylabel, save_path, info, overlay=None):
    values = _as_channels(window)
    overlay_values = _as_channels(overlay) if overlay is not None else None
    x_axis, xlabel = _time_axis(info, values.shape[0])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if values.shape[1] == 1:
        ax.plot(x_axis, values[:, 0], color="black", linewidth=0.9, label="window")
        if overlay_values is not None:
            ax.plot(
                x_axis,
                overlay_values[:, 0],
                color="tab:red",
                linewidth=1.3,
                label="exact clean injected component",
            )
    else:
        ax.plot(x_axis, values[:, 0], linewidth=0.9, label="I")
        ax.plot(x_axis, values[:, 1], linewidth=0.9, label="Q")
        ax.plot(
            x_axis,
            np.sqrt(np.sum(values[:, :2] ** 2, axis=1)),
            linewidth=1.1,
            label="magnitude",
        )
        if overlay_values is not None:
            ax.plot(
                x_axis,
                np.sqrt(np.sum(overlay_values[:, :2] ** 2, axis=1)),
                color="tab:red",
                linewidth=1.3,
                label="clean-signal magnitude",
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_overview(
    raw_clean,
    noise_window,
    signal_window,
    clean_norm,
    save_path,
    info,
    event_suffix,
):
    rows = [
        (
            f"Exact injected training-family amplitude\n{event_suffix}",
            raw_clean,
            None,
            "raw amplitude",
        ),
        ("Normalized noise-only window", noise_window, None, "normalized amplitude"),
        (
            f"Normalized noise + signal window\n{event_suffix}",
            signal_window,
            clean_norm,
            "normalized amplitude",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, (title, window, overlay, ylabel) in zip(axes, rows):
        values = _as_channels(window)
        overlay_values = _as_channels(overlay) if overlay is not None else None
        x_axis, xlabel = _time_axis(info, values.shape[0])

        if values.shape[1] == 1:
            ax.plot(x_axis, values[:, 0], color="black", linewidth=0.8, label="window")
            if overlay_values is not None:
                ax.plot(
                    x_axis,
                    overlay_values[:, 0],
                    color="tab:red",
                    linewidth=1.2,
                    label="exact clean injected component",
                )
        else:
            ax.plot(x_axis, values[:, 0], linewidth=0.8, label="I")
            ax.plot(x_axis, values[:, 1], linewidth=0.8, label="Q")
            if overlay_values is not None:
                ax.plot(
                    x_axis,
                    np.sqrt(np.sum(overlay_values[:, :2] ** 2, axis=1)),
                    color="tab:red",
                    linewidth=1.2,
                    label="clean-signal magnitude",
                )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_pretraining_sanity_plots(
    preprocessing_info,
    output_dir,
    split="val",
    strict=True,
    max_candidates=2048,
):
    """
    Save deterministic checks before training starts.

    The plots show the exact raw injected clean component, a representative
    normalized noise-only window, and a normalized noise+signal window with
    its aligned clean component overlaid.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train, val, test")

    os.makedirs(output_dir, exist_ok=True)
    noisy = _open_window_memmap(preprocessing_info, split, "norm")
    labels = _open_window_memmap(preprocessing_info, split, "lbl")
    clean_raw = _open_window_memmap(preprocessing_info, split, "clean_raw")
    clean_norm = _open_window_memmap(preprocessing_info, split, "clean_norm")

    missing = []
    if noisy is None:
        missing.append(f"{split}_norm_path")
    if labels is None:
        missing.append(f"{split}_lbl_path")
    if clean_raw is None:
        missing.append(f"{split}_clean_raw_path")
    if clean_norm is None:
        missing.append(f"{split}_clean_norm_path")
    if missing:
        message = (
            "Cannot create complete pre-training sanity plots; missing artifacts: "
            + ", ".join(missing)
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
        return {}

    label_values = np.asarray(labels, dtype=bool)
    noise_indices = np.flatnonzero(~label_values)
    signal_indices = np.flatnonzero(label_values)
    if noise_indices.size == 0 or signal_indices.size == 0:
        message = (
            f"Pre-training sanity check requires both classes in split='{split}', "
            f"but found noise={noise_indices.size}, signal={signal_indices.size}."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
        return {}

    noise_idx = _choose_representative_noise_index(
        noisy, noise_indices, max_candidates=max_candidates
    )
    isolated_candidates = _isolated_fully_contained_candidates(
        preprocessing_info,
        split,
        signal_indices,
        max_candidates=max_candidates,
    )
    if isolated_candidates.size:
        signal_idx = _choose_strong_signal_index(
            clean_norm,
            isolated_candidates,
            max_candidates=isolated_candidates.size,
        )
        signal_selection = "strongest isolated fully-contained signal"
    else:
        signal_idx = _choose_strong_signal_index(
            clean_norm, signal_indices, max_candidates=max_candidates
        )
        signal_selection = "strongest sampled signal; isolation unavailable"
        warnings.warn(
            "Could not find an isolated fully-contained signal window for the "
            "pre-training sanity plot. Falling back to the strongest sampled "
            "signal window, which may contain overlapping or clipped events."
        )

    selected_event = _selected_event_metadata(preprocessing_info, split, signal_idx)
    _validate_selected_event_alignment(
        clean_raw[signal_idx],
        selected_event,
        signal_selection,
    )
    event_suffix = _event_title_suffix(selected_event)
    overlap_paths = save_overlapping_injection_report(
        preprocessing_info,
        output_dir=output_dir,
    )

    paths = {
        "raw_clean_signal": os.path.join(output_dir, "raw_clean_signal.png"),
        "noise_only_window": os.path.join(output_dir, "noise_only_window.png"),
        "noise_plus_signal_window": os.path.join(
            output_dir, "noise_plus_signal_window.png"
        ),
        "overview": os.path.join(output_dir, "pretraining_sanity_overview.png"),
    }

    _plot_window(
        clean_raw[signal_idx],
        (
            "Exact injected training-family amplitude\n"
            f"{event_suffix} | {split} index={signal_idx}"
        ),
        "raw amplitude",
        paths["raw_clean_signal"],
        preprocessing_info,
    )
    _plot_window(
        noisy[noise_idx],
        f"Representative normalized noise-only window | {split} index={noise_idx}",
        "normalized amplitude",
        paths["noise_only_window"],
        preprocessing_info,
    )
    _plot_window(
        noisy[signal_idx],
        (
            f"Normalized noise + signal window\n{event_suffix} | "
            f"{split} index={signal_idx}"
        ),
        "normalized amplitude",
        paths["noise_plus_signal_window"],
        preprocessing_info,
        overlay=clean_norm[signal_idx],
    )
    _plot_overview(
        clean_raw[signal_idx],
        noisy[noise_idx],
        noisy[signal_idx],
        clean_norm[signal_idx],
        paths["overview"],
        preprocessing_info,
        event_suffix,
    )

    paths.update(overlap_paths)
    summary = {
        "split": split,
        "noise_index": noise_idx,
        "signal_index": signal_idx,
        "signal_selection": signal_selection,
        "selected_event": selected_event,
        "num_isolated_fully_contained_candidates": int(isolated_candidates.size),
        "num_noise_windows": int(noise_indices.size),
        "num_signal_windows": int(signal_indices.size),
        "signal_clean_raw_peak": float(np.max(np.abs(clean_raw[signal_idx]))),
        "signal_clean_normalized_peak": float(np.max(np.abs(clean_norm[signal_idx]))),
        "noise_window_mean": float(np.mean(noisy[noise_idx])),
        "noise_window_std": float(np.std(noisy[noise_idx])),
        "signal_window_mean": float(np.mean(noisy[signal_idx])),
        "signal_window_std": float(np.std(noisy[signal_idx])),
        "paths": paths,
    }
    summary_path = os.path.join(output_dir, "pretraining_sanity_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    paths["summary"] = summary_path

    print(
        "[PRETRAIN CHECK] Saved raw signal, noise-only, and noise+signal plots "
        f"under: {output_dir}"
    )
    return paths
