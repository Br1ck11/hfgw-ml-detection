import bisect
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import math
import os
import csv
from typing import Optional

from data_pre_processing.tiq_data_loader import load_tiq_data_segment
from data_pre_processing.window_data import window_segment
from data_pre_processing.stats import stream_welford_stats, stream_min_max, blockwise_normalize_to_path

from data_pre_processing.chirp_BW_conv_signal_generation import (
    warm_trimmed_cache,
    get_trimmed_waveform,
    clear_trimmed_cache
)


# --------------------------------------------------------------------- #
# Metadata CSV helpers (A2 / A3)
# --------------------------------------------------------------------- #

EVENT_METADATA_COLUMNS = [
    "split", "source_file", "event_id", "mass_solar", "target_peak_snr",
    "snr_peak", "snr_energy", "snr_energy_complex",
    "complex_changed_samples", "stored_changed_samples",
    "effective_stored_peak_snr",
    "injection_start_sample", "injection_end_sample", "trimmed_signal_length",
    "window_size", "step_size", "noise_mean", "noise_std",
    "response_mode", "chirp_mode", "sampling_rate",
]

WINDOW_METADATA_COLUMNS = [
    "split", "window_index", "window_start_sample", "window_end_sample",
    "event_id", "overlaps_signal", "overlap_fraction",
    "peak_snr_window", "energy_snr_window",
]


def _draw_non_overlapping_start(rng, reserved_starts, reserved_ends, sig_len,
                                segment_length, margin, max_attempts):
    """
    Draw a start index such that the support [t, t + sig_len) keeps at least
    `margin` samples of clear gap to every previously placed injection.

    With margin >= window_size, NO sliding window (of length window_size) can
    contain samples from two different events: a window can only bridge two
    events whose gap is smaller than the window length.

    `reserved_starts` / `reserved_ends` are parallel, sorted lists of the raw
    supports of already-placed injections; on success the new support is
    inserted and the start index returned, otherwise None (no free slot found
    after `max_attempts` rejection-sampling tries).
    """
    max_start = segment_length - sig_len
    if max_start < 0:
        return None
    for _ in range(int(max_attempts)):
        t = int(rng.integers(0, max_start + 1))
        lo = t - margin               # neighbor must end at or before this
        hi = t + sig_len + margin     # neighbor must start at or after this
        i = bisect.bisect_left(reserved_starts, hi)
        if i > 0 and reserved_ends[i - 1] > lo:
            continue
        reserved_starts.insert(i, t)
        reserved_ends.insert(i, t + sig_len)
        return t
    return None


def _append_rows_csv(path, columns, rows):
    """Append rows (list of dicts) to a CSV, writing the header once."""
    if not rows:
        return
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# --- Streaming TF datasets from memmaps (VECTORIZED FAST PATH) --- #

def make_train_dataset_from_memmap(
    data_path,
    labels_path,
    shape,
    dtype,
    batch_size=128,
    channels=1,
    shuffle_buffer=200_000,
    clean_path=None,
):
    """
    Streaming training dataset.

    If `clean_path` is given, every batch is a 3-tuple
    (x, y, clean_signal_window) where the clean windows are aligned
    one-to-one with the noisy windows (zeros for noise-only windows).
    Otherwise the legacy 2-tuple (x, y) is yielded.
    """
    X = np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
    Y = np.memmap(labels_path, mode="r", dtype=np.bool_, shape=(shape[0],))
    C = (
        np.memmap(clean_path, mode="r", dtype=dtype, shape=shape)
        if clean_path is not None else None
    )

    def _load_batch(batch_indices):
        idx = batch_indices.numpy()
        x = X[idx]
        y = Y[idx]
        if channels == 1 and x.ndim == 2:
            x = x[..., None]
        if C is None:
            return x, y
        c = C[idx]
        if channels == 1 and c.ndim == 2:
            c = c[..., None]
        return x, y, c

    ds = tf.data.Dataset.range(shape[0])
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size, drop_remainder=True)

    def _tf_load_batch(idx):
        if C is None:
            x, y = tf.py_function(
                _load_batch, [idx], [dtype, tf.bool]
            )
            x.set_shape((None, shape[1], channels))
            y.set_shape((None,))
            y = tf.cast(y, tf.float32)
            return x, y
        x, y, c = tf.py_function(
            _load_batch, [idx], [dtype, tf.bool, dtype]
        )
        x.set_shape((None, shape[1], channels))
        y.set_shape((None,))
        c.set_shape((None, shape[1], channels))
        y = tf.cast(y, tf.float32)
        return x, y, c

    ds = ds.map(
        _tf_load_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds #.repeat() only use repeat if your data set is too small and you wanna repeat samples to fill up a certain steps per epoch


def make_eval_dataset_from_memmap(
    data_path,
    labels_path,
    shape,
    dtype,
    batch_size=128,
    channels=1,
    clean_path=None,
):
    X = np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
    Y = np.memmap(labels_path, mode="r", dtype=np.bool_, shape=(shape[0],))
    C = (
        np.memmap(clean_path, mode="r", dtype=dtype, shape=shape)
        if clean_path is not None else None
    )

    def _load_batch(batch_indices):
        idx = batch_indices.numpy()
        x = X[idx]
        y = Y[idx]
        if channels == 1 and x.ndim == 2:
            x = x[..., None]
        if C is None:
            return x, y   # <-- bool labels here
        c = C[idx]
        if channels == 1 and c.ndim == 2:
            c = c[..., None]
        return x, y, c

    ds = tf.data.Dataset.range(shape[0])
    ds = ds.batch(batch_size, drop_remainder=True)

    def _tf_load_batch(idx):
        if C is None:
            x, y = tf.py_function(
                _load_batch, [idx], [dtype, tf.bool]
            )
            x.set_shape((None, shape[1], channels))
            y.set_shape((None,))
            y = tf.cast(y, tf.float32)   # <-- cast HERE
            return x, y
        x, y, c = tf.py_function(
            _load_batch, [idx], [dtype, tf.bool, dtype]
        )
        x.set_shape((None, shape[1], channels))
        y.set_shape((None,))
        c.set_shape((None, shape[1], channels))
        y = tf.cast(y, tf.float32)
        return x, y, c

    ds = ds.map(
        _tf_load_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
    

def pre_processing_with_memmap(
    # IMPORTANT INVARIANT:
    # global_mean_input / global_std_input are GLOBAL TRAINING statistics.
    # They are used consistently for:
    #   (1) z-score normalization
    #   (2) SNR-based signal amplitude scaling
    # No per-file or per-segment statistics are allowed.
    filepath_suffixes, filepath_template, num_samples_to_read_per_file, offset,
    window_size, step_size, train_ratio, val_ratio, test_ratio, dtype,
    normalization_type='min_max',
    global_min_input=None,
    global_max_input=None,
    global_mean_input=None,
    global_std_input=None,
    calculate_stats=True,
    
    # --- Mode Selection ---
    use_amps=True,    # Default legacy mode
    use_I_Q=False,    # New raw mode
    
    # --- Injection Parameters ---
    inject_signals=False,
    signal_injection_probability=1.0,
    m_PBH_injection_list=None,
    amplitude_spectrum_range=None,
    num_signals_to_inject_per_segment=1,
    snr_based_injection=False,
    custom_noise_std=None,
    # --- No-overlap injection mode ---
    # When True, injected signals are placed so that the gap between any two
    # injected supports is at least `no_overlap_margin_samples`. With the
    # default margin (None -> window_size) NO sliding window can contain
    # samples from two different events, because a window of length W can
    # only bridge two events whose gap is < W. Placement uses rejection
    # sampling; injections that cannot be placed after
    # `no_overlap_max_attempts` tries are skipped (counted + reported).
    no_overlap_injections=False,
    no_overlap_margin_samples=None,
    no_overlap_max_attempts=200,
    # ----------------------------
    f0_gw=5.0e9, Gamma_gw=100e3, N_gw=32768, M_solar=1.988e30,
    # --- Cavity response (A5) ---
    # "real_lorentzian" reproduces the existing behavior exactly.
    # "complex_breit_wigner" uses the complex single-pole response.
    response_mode='real_lorentzian',
    # --- Clean-signal saving & metadata (A1-A4) ---
    save_clean_signals=True,        # save exact injected (trimmed) clean windows
    save_metadata=True,             # write event/window metadata CSVs
    include_clean_in_datasets=False,  # tf datasets yield (x, y, clean) 3-tuples
    memmap_dir='./memmaps',
    stats_dir=None,
    return_tf_datasets: bool = True,
    tf_batch_size: int = 512,
    tf_shuffle: bool = True,
    tf_repeat: bool = False,
    random_seed: Optional[int] = None,
    reject_unrepresentable_injections: bool = True,
    return_info: bool = False,      # additionally return an info dict (paths etc.)
):
    # --- 1. MODE VALIDATION --- #
    if not use_amps and not use_I_Q:
        raise ValueError("Configuration Error: You must select a mode of operation. Set either 'use_amps=True' or 'use_I_Q=True'.")
    
    if use_amps and use_I_Q:
        raise ValueError("Configuration Error: Ambiguous mode. Please set ONLY ONE to True (use_amps OR use_I_Q), not both.")
        
    # Determine channels for final shape
    # If Amps: (N, Window) -> TF expands to (N, Window, 1)
    # If I/Q:  (N, Window, 2) -> TF keeps (N, Window, 2)
    num_channels = 2 if use_I_Q else 1
    
    if stats_dir is None:
        stats_dir = memmap_dir

    # I/Q normalization uses the component-wise mean/std, while peak-SNR
    # scaling uses std(|noise|). An explicit custom_noise_std fully specifies
    # the latter and must not require a saved global_std_mag.npy file.
    global_std_mag = None
    iq_snr_std_source = None
    if use_I_Q and custom_noise_std is not None:
        global_std_mag = float(custom_noise_std)
        iq_snr_std_source = "explicit custom_noise_std"
    elif use_I_Q and not calculate_stats:
        std_mag_path = os.path.join(stats_dir, "global_std_mag.npy")
        if os.path.isfile(std_mag_path):
            global_std_mag = float(np.load(std_mag_path))
            iq_snr_std_source = std_mag_path
        elif inject_signals and snr_based_injection:
            raise FileNotFoundError(
                "I/Q SNR-based injection requires std(|noise|). Provide "
                "custom_noise_std explicitly or supply the saved file "
                f"'{std_mag_path}'. The I/Q normalization std is not generally "
                "equal to std(|noise|)."
            )
    
    print(f"--- Preprocessing Mode: {'I/Q (2 Channels)' if use_I_Q else 'Amplitude (1 Channel)'} ---")
    if inject_signals:
        policy = "STOP" if reject_unrepresentable_injections else "WARN AND CONTINUE"
        print(
            "[INJECTION VALIDITY] Zero-change stored injections: "
            f"{policy}. Set reject_unrepresentable_injections="
            f"{not reject_unrepresentable_injections} to choose the other behavior."
        )

    # ------------------------------------------------------------------
    # AUTOMATIC PRE-PASS: compute GLOBAL training mean/std if needed
    # ------------------------------------------------------------------
    if inject_signals and snr_based_injection and calculate_stats:
        print(
            "\n[INFO] SNR-based injection requires GLOBAL training statistics.\n"
            "[INFO] Running automatic stats-only pre-pass (NO injection)...\n"
        )

        _ = pre_processing_with_memmap(
            filepath_suffixes=filepath_suffixes,
            filepath_template=filepath_template,
            num_samples_to_read_per_file=num_samples_to_read_per_file,
            offset=offset,
            window_size=window_size,
            step_size=step_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            dtype=dtype,
            normalization_type=normalization_type,
            calculate_stats=True,
            inject_signals=False,
            snr_based_injection=False,
            use_amps=use_amps,
            use_I_Q=use_I_Q,
            save_clean_signals=False,
            save_metadata=False,
            memmap_dir=memmap_dir,
            stats_dir=stats_dir,
            random_seed=random_seed,
            reject_unrepresentable_injections=reject_unrepresentable_injections,
            return_tf_datasets=False
        )

        # Load the computed stats from disk (written at end of function)
        global_mean = np.load(os.path.join(stats_dir, "global_mean.npy"))
        global_std  = np.load(os.path.join(stats_dir, "global_std.npy"))

        # Load GLOBAL magnitude std as well for I/Q SNR definition, unless an
        # explicit custom SNR scale was supplied by the caller.
        if use_I_Q and custom_noise_std is None:
            global_std_mag = float(
                np.load(os.path.join(stats_dir, "global_std_mag.npy"))
            )
            iq_snr_std_source = os.path.join(stats_dir, "global_std_mag.npy")

        # Freeze stats for the real pass
        global_mean_input = global_mean
        global_std_input  = global_std
        calculate_stats   = False

        print(
            f"[INFO] Using GLOBAL std = {global_std:.4e} and mean = {global_mean:.4e} "
            "for SNR-based injection.\n"
        )

    # --- PASS 0: Cache Warming --- #
    if inject_signals and m_PBH_injection_list and amplitude_spectrum_range:
        # ... (Cache warming logic remains the same) ...
        if snr_based_injection:
            print(f"--- Warming Signal Cache (SNR Mode, response_mode={response_mode}) ---")
            warm_trimmed_cache(m_PBH_injection_list, [1.0], f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)
        else:
            print(f"--- Warming Signal Cache (Raw Amp Mode, response_mode={response_mode}) ---")
            warm_trimmed_cache(m_PBH_injection_list, amplitude_spectrum_range, f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)

    # --- Pre-Pass: Normalize Injection Counts --- #
    injection_counts = {}
    if isinstance(num_signals_to_inject_per_segment, int):
        injection_counts = {'train': num_signals_to_inject_per_segment, 'val': num_signals_to_inject_per_segment, 'test': num_signals_to_inject_per_segment}
    elif isinstance(num_signals_to_inject_per_segment, dict):
        injection_counts = injection_counts = num_signals_to_inject_per_segment
    else:
        injection_counts = {'train': 1, 'val': 1, 'test': 1}

    actual_injected_counts = {'train': 0, 'val': 0, 'test': 0}
    skipped_no_overlap_counts = {'train': 0, 'val': 0, 'test': 0}
    no_overlap_margin = (
        int(no_overlap_margin_samples)
        if no_overlap_margin_samples is not None else int(window_size)
    )

    # --- PASS 1: Pre-calculation of total chunk counts --- #
    print("--- Pass 1: Calculating total chunk sizes ---")
    total_train_chunks = 0; total_val_chunks = 0; total_test_chunks = 0

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-9):
        raise ValueError(f"Error: Ratios do not sum to 1.0! Sum is {train_ratio + val_ratio + test_ratio}.")

    for suffix_string in filepath_suffixes:
        file_data_length = num_samples_to_read_per_file
        train_len = int(file_data_length * train_ratio)
        val_len = int(file_data_length * val_ratio)
        test_len = int(file_data_length * test_ratio)
        
        if train_len >= window_size: total_train_chunks += (train_len - window_size) // step_size + 1
        if val_len >= window_size:   total_val_chunks += (val_len - window_size) // step_size + 1
        if test_len >= window_size:  total_test_chunks += (test_len - window_size) // step_size + 1

    print(f"Pre-calculated totals: Train={total_train_chunks}, Val={total_val_chunks}, Test={total_test_chunks}")

    # --- Pre-allocation of final NumPy arrays --- #
    print("\n--- Pre-allocating final arrays ---")
    os.makedirs(memmap_dir, exist_ok=True)
    
    # Define shapes based on mode
    # Note: For Amps, we keep it 2D (N, W). For I/Q, we need 3D (N, W, 2).
    if use_I_Q:
        memmap_shape_train = (total_train_chunks, window_size, 2)
        memmap_shape_val   = (total_val_chunks, window_size, 2)
        memmap_shape_test  = (total_test_chunks, window_size, 2)
    else:
        memmap_shape_train = (total_train_chunks, window_size)
        memmap_shape_val   = (total_val_chunks, window_size)
        memmap_shape_test  = (total_test_chunks, window_size)

    train_path = os.path.join(memmap_dir, f"train_chunks_{total_train_chunks}x{window_size}_{'IQ' if use_I_Q else 'Amp'}.dat")
    val_path   = os.path.join(memmap_dir, f"val_chunks_{total_val_chunks}x{window_size}_{'IQ' if use_I_Q else 'Amp'}.dat")
    test_path  = os.path.join(memmap_dir, f"test_chunks_{total_test_chunks}x{window_size}_{'IQ' if use_I_Q else 'Amp'}.dat")

    train_lbl_path = os.path.join(memmap_dir, f"train_labels_{total_train_chunks}.dat")
    val_lbl_path   = os.path.join(memmap_dir, f"val_labels_{total_val_chunks}.dat")
    test_lbl_path  = os.path.join(memmap_dir, f"test_labels_{total_test_chunks}.dat")

    # --- SNR memmap paths and arrays ---
    train_snr_path = os.path.join(memmap_dir, f"train_snr_{total_train_chunks}.dat")
    val_snr_path   = os.path.join(memmap_dir, f"val_snr_{total_val_chunks}.dat")
    test_snr_path  = os.path.join(memmap_dir, f"test_snr_{total_test_chunks}.dat")

    final_train_chunks_raw = np.memmap(train_path, mode='w+', dtype=dtype, shape=memmap_shape_train)
    final_val_chunks_raw   = np.memmap(val_path,   mode='w+', dtype=dtype, shape=memmap_shape_val)
    final_test_chunks_raw  = np.memmap(test_path,  mode='w+', dtype=dtype, shape=memmap_shape_test)

    final_train_labels = np.memmap(train_lbl_path, mode='w+', dtype=np.bool_, shape=(total_train_chunks,))
    final_val_labels   = np.memmap(val_lbl_path,   mode='w+', dtype=np.bool_, shape=(total_val_chunks,))
    final_test_labels  = np.memmap(test_lbl_path,  mode='w+', dtype=np.bool_, shape=(total_test_chunks,))

    final_train_snr = np.memmap(train_snr_path, mode='w+', dtype=np.float32, shape=(total_train_chunks,))
    final_val_snr   = np.memmap(val_snr_path,   mode='w+', dtype=np.float32, shape=(total_val_chunks,))
    final_test_snr  = np.memmap(test_snr_path,  mode='w+', dtype=np.float32, shape=(total_test_chunks,))

    # Initialize
    final_train_chunks_raw[:] = 0; final_val_chunks_raw[:] = 0; final_test_chunks_raw[:] = 0
    final_train_labels[:] = False; final_val_labels[:] = False; final_test_labels[:] = False
    final_train_snr[:] = 0.0; final_val_snr[:] = 0.0; final_test_snr[:] = 0.0

    # --- A1: Clean injected-signal windows (exact injected component) --- #
    # Only meaningful when signals are injected. Aligned 1:1 with the noisy
    # windows; zeros everywhere except at the injected trimmed waveform.
    save_clean_signals = bool(save_clean_signals and inject_signals)
    # Metadata only makes sense for injection runs (noise-only windows of those
    # runs are still fully recorded with event_id = -1).
    save_metadata = bool(save_metadata and inject_signals)
    clean_arrays = {'train': None, 'val': None, 'test': None}
    clean_raw_paths = {'train': None, 'val': None, 'test': None}
    if save_clean_signals:
        clean_raw_paths = {
            'train': os.path.join(memmap_dir, f"train_clean_{total_train_chunks}x{window_size}_{'IQ' if use_I_Q else 'Amp'}.dat"),
            'val':   os.path.join(memmap_dir, f"val_clean_{total_val_chunks}x{window_size}_{'IQ' if use_I_Q else 'Amp'}.dat"),
            'test':  os.path.join(memmap_dir, f"test_clean_{total_test_chunks}x{window_size}_{'IQ' if use_I_Q else 'Amp'}.dat"),
        }
        clean_arrays = {
            'train': np.memmap(clean_raw_paths['train'], mode='w+', dtype=dtype, shape=memmap_shape_train),
            'val':   np.memmap(clean_raw_paths['val'],   mode='w+', dtype=dtype, shape=memmap_shape_val),
            'test':  np.memmap(clean_raw_paths['test'],  mode='w+', dtype=dtype, shape=memmap_shape_test),
        }
        for arr in clean_arrays.values():
            arr[:] = 0

    # --- A2 / A3: Metadata CSV paths (fresh per preprocessing run) --- #
    event_metadata_path = os.path.join(memmap_dir, "event_metadata.csv")
    window_metadata_path = os.path.join(memmap_dir, "window_metadata.csv")
    if save_metadata:
        for _p in (event_metadata_path, window_metadata_path):
            if os.path.exists(_p):
                os.remove(_p)
    event_id_counter = 0

    train_idx_counter = 0; val_idx_counter = 0; test_idx_counter = 0

    # --- PASS 2: Processing files --- #
    print("\n--- Pass 2: Processing files and filling arrays ---")
    rng = np.random.default_rng(random_seed)

    num_files_loaded = 0
    for i, suffix_string in enumerate(filepath_suffixes):
        current_filepath = filepath_template.format(suffix_string)
        file_label = f"File {i+1} ({current_filepath})"
        print(f"\nProcessing {file_label}...")
        iq_channel_data, fs_val = load_tiq_data_segment(current_filepath, offset, num_samples_to_read_per_file)
        # FAIL LOUDLY on unreadable data. A failed file used to be silently
        # skipped, which left the memmaps all-zero and produced the classic
        # "0 injected signals / mean 0 / std 0" failure downstream.
        if iq_channel_data is None:
            if not os.path.isfile(current_filepath):
                raise FileNotFoundError(
                    f"Data file not found: '{os.path.abspath(current_filepath)}'.\n"
                    f"filepath_template='{filepath_template}', suffix='{suffix_string}', "
                    f"cwd='{os.getcwd()}'.\n"
                    "Relative templates resolve against the CURRENT WORKING "
                    "DIRECTORY — run from the project root or use an absolute "
                    "path (the training scripts anchor it to _PROJECT_ROOT)."
                )
            raise IOError(
                f"Failed to load tiq data from '{current_filepath}' "
                "(loader returned None — corrupt file or wrong format?)."
            )
        num_files_loaded += 1

        file_data_length = len(iq_channel_data)
        train_len = int(file_data_length * train_ratio)
        val_len = int(file_data_length * val_ratio)

        segments_to_process = {
            'train': (iq_channel_data[0:train_len], final_train_chunks_raw, final_train_labels, final_train_snr, train_idx_counter),
            'val': (iq_channel_data[train_len : train_len + val_len], final_val_chunks_raw, final_val_labels, final_val_snr, val_idx_counter),
            'test': (iq_channel_data[train_len + val_len :], final_test_chunks_raw, final_test_labels, final_test_snr, test_idx_counter),
        }

        for segment_type, (raw_segment_complex, final_chunks_arr, final_labels_arr, final_snr_arr, start_idx) in segments_to_process.items():
            if len(raw_segment_complex) < window_size: continue
            segment_length = len(raw_segment_complex)
            has_signal_flags_for_segment = np.zeros(segment_length, dtype=bool)

            # Clean injected-signal component for this segment (A1):
            # zeros everywhere except where trimmed waveforms are injected.
            need_clean_segment = inject_signals and (
                save_clean_signals or save_metadata or snr_based_injection
            )
            clean_segment_complex = (
                np.zeros(segment_length, dtype=np.complex64)
                if need_clean_segment else None
            )
            events_this_segment = []  # (event_id, t_start, t_end, sig_len)
            # No-overlap mode: sorted supports of already-placed injections
            reserved_starts, reserved_ends = [], []

            # --- SNR Calc (GLOBAL TRAINING STD ONLY) --- #
            if snr_based_injection and inject_signals:
                if custom_noise_std is not None:
                    current_noise_std = float(custom_noise_std)
                    print(f"Using custom noise std = {current_noise_std:.3e}")
                else:
                    if global_std_input is None:
                        raise ValueError(
                            "Global training std must be provided for SNR-based injection. "
                            "Run a first pass with calculate_stats=True on TRAINING data."
                        )
                    if use_I_Q:
                        current_noise_std = float(global_std_mag)
                    else:
                        current_noise_std = float(global_std_input)
                    if use_I_Q:
                        print(
                            f"[INFO][SNR] Using MAGNITUDE-based noise std for SNR injection: "
                            f"std(|n|) = {current_noise_std:.4e} | "
                            f"Normalization uses I/Q mean = {global_mean_input:.4e}, "
                            f"I/Q std = {global_std_input:.4e}"
                        )
                    else:
                        print(
                            f"[INFO][SNR] Using AMPLITUDE-based noise std for SNR injection: "
                            f"std(|n|) = {current_noise_std:.4e}"
                        )
                            
            # --- Injection --- #
            if inject_signals and m_PBH_injection_list and amplitude_spectrum_range:
                current_injection_count = injection_counts.get(segment_type, 0)
                if current_injection_count > 0:
                    # Noise std definitions used purely for SNR *metadata* (A4).
                    # The main per-window SNR definition (peak SNR) is unchanged.
                    _noise_std_meta = (
                        float(current_noise_std)
                        if (snr_based_injection and inject_signals) else float('nan')
                    )
                    _noise_mean_meta = (
                        float(global_mean_input)
                        if global_mean_input is not None else float('nan')
                    )
                    _noise_std_complex_meta = (
                        float(global_std_input)
                        if (use_I_Q and global_std_input is not None) else float('nan')
                    )

                    event_rows = []
                    for _ in range(current_injection_count):
                        if rng.random() < signal_injection_probability:
                            m_pbh = rng.choice(m_PBH_injection_list)
                            target_snr = float('nan')
                            if snr_based_injection:
                                target_snr = rng.choice(amplitude_spectrum_range)
                                waveform = get_trimmed_waveform(m_pbh, 1.0, f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)
                                if waveform.size > 0:
                                    peak = np.max(np.abs(waveform))
                                    if peak > 0: waveform *= (target_snr * current_noise_std / peak)
                            else:
                                amp_val = rng.choice(amplitude_spectrum_range)
                                waveform = get_trimmed_waveform(m_pbh, amp_val, f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)

                            sig_len = waveform.size
                            if sig_len > 0 and sig_len <= segment_length:
                                if no_overlap_injections:
                                    t_start = _draw_non_overlapping_start(
                                        rng, reserved_starts, reserved_ends,
                                        sig_len, segment_length,
                                        no_overlap_margin,
                                        no_overlap_max_attempts,
                                    )
                                    if t_start is None:
                                        skipped_no_overlap_counts[segment_type] += 1
                                        continue
                                else:
                                    max_start_index = segment_length - sig_len
                                    t_start = rng.integers(0, max_start_index + 1)
                                before_injection = raw_segment_complex[
                                    t_start : t_start + sig_len
                                ].copy()
                                raw_segment_complex[t_start : t_start + sig_len] += waveform
                                after_injection = raw_segment_complex[
                                    t_start : t_start + sig_len
                                ]

                                complex_changed_samples = int(
                                    np.count_nonzero(after_injection != before_injection)
                                )
                                if use_amps:
                                    before_stored = np.abs(before_injection).astype(dtype)
                                    after_stored = np.abs(after_injection).astype(dtype)
                                    stored_delta = np.abs(after_stored - before_stored)
                                    stored_changed_samples = int(
                                        np.count_nonzero(after_stored != before_stored)
                                    )
                                else:
                                    before_stored = np.stack(
                                        [
                                            np.real(before_injection),
                                            np.imag(before_injection),
                                        ],
                                        axis=-1,
                                    ).astype(dtype)
                                    after_stored = np.stack(
                                        [
                                            np.real(after_injection),
                                            np.imag(after_injection),
                                        ],
                                        axis=-1,
                                    ).astype(dtype)
                                    stored_delta = np.sqrt(
                                        np.sum(
                                            np.square(after_stored - before_stored),
                                            axis=-1,
                                        )
                                    )
                                    stored_changed_samples = int(
                                        np.count_nonzero(
                                            np.any(after_stored != before_stored, axis=-1)
                                        )
                                    )
                                effective_stored_peak_snr = (
                                    float(np.max(stored_delta) / current_noise_std)
                                    if snr_based_injection and stored_delta.size
                                    else float("nan")
                                )
                                if snr_based_injection and stored_changed_samples == 0:
                                    message = (
                                        "Target SNR "
                                        f"{target_snr:.3e} changed zero stored "
                                        f"{np.dtype(dtype).name} samples. This "
                                        "injection is absent from the model input."
                                    )
                                    if reject_unrepresentable_injections:
                                        raise ValueError(
                                            f"[INVALID INJECTION] {message}"
                                        )
                                    print(f"[WARNING][INJECTION] {message}")

                                has_signal_flags_for_segment[t_start : t_start + sig_len] = True
                                actual_injected_counts[segment_type] += 1

                                # --- A1: record the EXACT injected component --- #
                                if clean_segment_complex is not None:
                                    clean_segment_complex[t_start : t_start + sig_len] += waveform.astype(np.complex64)

                                # --- A2 / A4: event metadata --- #
                                events_this_segment.append(
                                    (event_id_counter, int(t_start), int(t_start + sig_len), int(sig_len))
                                )
                                if save_metadata:
                                    abs_wave = np.abs(waveform)
                                    energy_raw = float(np.sqrt(np.sum(abs_wave ** 2)))
                                    snr_peak_meta = float(np.max(abs_wave) / _noise_std_meta) if _noise_std_meta == _noise_std_meta else float('nan')
                                    snr_energy_meta = energy_raw / _noise_std_meta if _noise_std_meta == _noise_std_meta else float('nan')
                                    snr_energy_complex_meta = (
                                        energy_raw / _noise_std_complex_meta
                                        if _noise_std_complex_meta == _noise_std_complex_meta else float('nan')
                                    )
                                    event_rows.append({
                                        "split": segment_type,
                                        "source_file": suffix_string,
                                        "event_id": event_id_counter,
                                        "mass_solar": float(m_pbh) / float(M_solar),
                                        "target_peak_snr": target_snr,
                                        "snr_peak": snr_peak_meta,
                                        "snr_energy": snr_energy_meta,
                                        "snr_energy_complex": snr_energy_complex_meta,
                                        "complex_changed_samples": complex_changed_samples,
                                        "stored_changed_samples": stored_changed_samples,
                                        "effective_stored_peak_snr": effective_stored_peak_snr,
                                        "injection_start_sample": int(t_start),
                                        "injection_end_sample": int(t_start + sig_len),
                                        "trimmed_signal_length": int(sig_len),
                                        "window_size": int(window_size),
                                        "step_size": int(step_size),
                                        "noise_mean": _noise_mean_meta,
                                        "noise_std": _noise_std_meta,
                                        "response_mode": response_mode,
                                        "chirp_mode": "spa_chirp",
                                        "sampling_rate": float(fs_val) if fs_val is not None else float('nan'),
                                    })
                                event_id_counter += 1

                    if save_metadata and event_rows:
                        _append_rows_csv(event_metadata_path, EVENT_METADATA_COLUMNS, event_rows)

            # --- LOGIC SPLIT: I/Q vs Amplitude --- #
            current_chunks = None

            if use_I_Q:
                # 1. Stack Real and Imaginary parts: Shape (Length, 2)
                complex_stacked = np.stack([np.real(raw_segment_complex), np.imag(raw_segment_complex)], axis=-1)

                # 2. Windowing for 2D data
                windows_view = sliding_window_view(complex_stacked, window_size, axis=0)[::step_size]
                current_chunks = windows_view.transpose(0, 2, 1)

            elif use_amps:
                amplitude_data = np.abs(raw_segment_complex)
                current_chunks = window_segment(amplitude_data, window_size, step_size)

            num_new_chunks = len(current_chunks)
            if num_new_chunks == 0: continue

            # --- Allocate SNR array for this segment ---
            chunk_snr_for_segment = np.zeros(num_new_chunks, dtype=np.float32)

            # --- Labels --- #
            if len(has_signal_flags_for_segment) >= window_size:
                chunk_labels_for_segment = sliding_window_view(has_signal_flags_for_segment, window_shape=window_size)[::step_size].any(axis=1)
            else:
                chunk_labels_for_segment = np.zeros(num_new_chunks, dtype=bool)

            # --- Compute per-window injected-signal peak SNR from clean component --- #
            if snr_based_injection and inject_signals:
                for w in range(num_new_chunks):
                    if chunk_labels_for_segment[w]:
                        start = w * step_size
                        end = start + window_size
                        if clean_segment_complex is not None:
                            sig_vals = clean_segment_complex[start:end]
                            chunk_snr_for_segment[w] = (
                                np.max(np.abs(sig_vals)) / current_noise_std
                            )
                        else:
                            raise RuntimeError(
                                "Exact clean component is required to compute "
                                "per-window injected-signal SNR."
                            )
                    else:
                        chunk_snr_for_segment[w] = 0.0
            else:
                # no injection → define SNR = 0 for all windows
                chunk_snr_for_segment[:] = 0.0

            # --- A1: window the clean injected component identically --- #
            if save_clean_signals and clean_segment_complex is not None:
                if use_I_Q:
                    clean_stacked = np.stack(
                        [np.real(clean_segment_complex), np.imag(clean_segment_complex)],
                        axis=-1,
                    )
                    clean_chunks = sliding_window_view(
                        clean_stacked, window_size, axis=0
                    )[::step_size].transpose(0, 2, 1)
                else:
                    clean_amp_segment = np.abs(clean_segment_complex)
                    clean_chunks = window_segment(clean_amp_segment, window_size, step_size)
                clean_arrays[segment_type][start_idx:start_idx + num_new_chunks] = clean_chunks[:num_new_chunks]

            # --- A3 / A4: per-window metadata --- #
            if save_metadata:
                _w_noise_std = (
                    float(current_noise_std)
                    if (snr_based_injection and inject_signals) else float('nan')
                )
                window_starts = np.arange(num_new_chunks, dtype=np.int64) * step_size
                window_ends = window_starts + window_size
                event_ids_per_window = np.full(num_new_chunks, -1, dtype=np.int64)
                overlap_fraction = np.zeros(num_new_chunks, dtype=np.float64)
                peak_snr_window = np.zeros(num_new_chunks, dtype=np.float64)
                energy_snr_window = np.zeros(num_new_chunks, dtype=np.float64)

                if clean_segment_complex is not None and events_this_segment:
                    clean_amp = np.abs(clean_segment_complex).astype(np.float64)
                    cs2 = np.concatenate(([0.0], np.cumsum(clean_amp ** 2)))
                    energy_snr_window = np.sqrt(
                        cs2[window_ends] - cs2[window_starts]
                    ) / _w_noise_std

                    for (eid, t0, t1, slen) in events_this_segment:
                        w_first = max(0, (t0 - window_size) // step_size + 1)
                        w_last = min(num_new_chunks - 1, (t1 - 1) // step_size)
                        if w_last < w_first:
                            continue
                        sel = slice(w_first, w_last + 1)
                        ws = window_starts[sel]
                        ov = np.minimum(ws + window_size, t1) - np.maximum(ws, t0)
                        ov = np.clip(ov, 0, None).astype(np.float64)
                        frac = ov / float(slen)
                        better = frac > overlap_fraction[sel]
                        event_ids_per_window[sel][better] = eid
                        overlap_fraction[sel] = np.maximum(overlap_fraction[sel], frac)
                        # Exact per-window peak SNR from the CLEAN component
                        for w in range(w_first, w_last + 1):
                            seg = clean_amp[window_starts[w]:window_ends[w]]
                            pk = seg.max() / _w_noise_std
                            if pk > peak_snr_window[w]:
                                peak_snr_window[w] = pk

                write_header = not os.path.exists(window_metadata_path)
                with open(window_metadata_path, "a", newline="") as _fh:
                    _writer = csv.writer(_fh)
                    if write_header:
                        _writer.writerow(WINDOW_METADATA_COLUMNS)
                    _writer.writerows(
                        (
                            segment_type,
                            int(start_idx + w),
                            int(window_starts[w]),
                            int(window_ends[w]),
                            int(event_ids_per_window[w]),
                            bool(chunk_labels_for_segment[w]),
                            float(overlap_fraction[w]),
                            float(peak_snr_window[w]),
                            float(energy_snr_window[w]),
                        )
                        for w in range(num_new_chunks)
                    )

            # --- Fill --- #
            end_idx = start_idx + num_new_chunks
            final_chunks_arr[start_idx:end_idx] = current_chunks
            final_labels_arr[start_idx:end_idx] = chunk_labels_for_segment
            final_snr_arr[start_idx:end_idx] = chunk_snr_for_segment

            if segment_type == 'train': train_idx_counter += num_new_chunks
            elif segment_type == 'val': val_idx_counter += num_new_chunks
            elif segment_type == 'test': test_idx_counter += num_new_chunks

        print(f"Cumulative chunks: Train={train_idx_counter}, Val={val_idx_counter}, Test={test_idx_counter}")

    # --- Hard sanity checks: never continue on an empty pipeline --- #
    if num_files_loaded == 0:
        raise RuntimeError(
            "No data files were loaded — every entry of filepath_suffixes "
            f"{filepath_suffixes} failed for template '{filepath_template}' "
            f"(cwd='{os.getcwd()}')."
        )
    if inject_signals and sum(actual_injected_counts.values()) == 0 and any(
        injection_counts.get(s, 0) > 0 for s in ("train", "val", "test")
    ):
        raise RuntimeError(
            "Signal injection was requested but 0 signals were injected "
            f"(requested per segment: {injection_counts}). This usually means "
            "the trimmed waveform is longer than a data segment or the data "
            "segments are empty."
        )

    # --- Cleanup --- #
    if inject_signals:
        clear_trimmed_cache()
        print("\nSIGNAL INJECTION REPORT: ", actual_injected_counts)
        if no_overlap_injections:
            total_skipped = sum(skipped_no_overlap_counts.values())
            print(
                "NO-OVERLAP MODE: margin = "
                f"{no_overlap_margin} samples "
                f"({'no two events can share a window' if no_overlap_margin >= window_size else 'events may still share windows (margin < window_size)'}) | "
                f"skipped (no free slot): {skipped_no_overlap_counts}"
            )
            if total_skipped > 0:
                total_placed = sum(actual_injected_counts.values())
                frac = total_skipped / max(total_placed + total_skipped, 1)
                print(
                    f"[WARNING][NO-OVERLAP] {total_skipped} injections "
                    f"({100 * frac:.1f}%) could not be placed. The segment is "
                    "too densely packed for the requested margin — reduce "
                    "num_signals_to_inject_per_segment, the margin, or use "
                    "more data."
                )

    # --- Normalization --- #
    print("\n--- Normalization ---")
    final_train_chunks_raw.flush(); final_val_chunks_raw.flush(); final_test_chunks_raw.flush()
    final_train_snr.flush(); final_val_snr.flush(); final_test_snr.flush()
    if save_clean_signals:
        for arr in clean_arrays.values():
            arr.flush()

    norm_params = {}
    if normalization_type == 'zscore':
        if calculate_stats:
            print("Computing GLOBAL training mean/std for normalization...")
            global_mean, global_std = stream_welford_stats(final_train_chunks_raw)
            # Persist GLOBAL stats for reuse (required for SNR-based injection)
            np.save(os.path.join(stats_dir, "global_mean.npy"), global_mean)
            np.save(os.path.join(stats_dir, "global_std.npy"),  global_std)
            
            # --- Additional GLOBAL magnitude std for I/Q SNR definition --- #
            if use_I_Q:
                print("Computing GLOBAL magnitude std for I/Q SNR definition...")
                mag_view = np.sqrt(
                    final_train_chunks_raw[..., 0]**2 +
                    final_train_chunks_raw[..., 1]**2
                )
                global_std_mag = float(np.std(mag_view))
                np.save(os.path.join(stats_dir, "global_std_mag.npy"), global_std_mag)
                iq_snr_std_source = os.path.join(stats_dir, "global_std_mag.npy")
        else:
            if global_mean_input is None or global_std_input is None:
                raise ValueError(
                    "Global mean/std must be provided when calculate_stats=False"
                )
            global_mean = float(global_mean_input)
            global_std  = float(global_std_input)
            print(
                f"[INFO][NORM] Using PRECOMPUTED normalization stats: "
                f"mean = {global_mean:.4e}, std = {global_std:.4e}"
            )

            if use_I_Q and global_std_mag is not None:
                print(
                    f"[INFO][NORM] Using I/Q peak-SNR noise scale from "
                    f"{iq_snr_std_source}: std(|n|) = {global_std_mag:.4e}"
                )
        norm_params = {'mean_value': global_mean, 'std_dev_value': global_std}
        print(f"Using zscore normalization with std. deviation = {global_std:.4e} and mean = {global_mean:.4e}")
    elif normalization_type == 'min_max':
        if calculate_stats is True:
            global_min, global_max = stream_min_max(final_train_chunks_raw)
        else:
            global_min = global_min_input
            global_max = global_max_input

        norm_params = {'min_value': global_min, 'max_value': global_max}
        print(f"Using min-max normalization with min = {global_min:.4e} and max = {global_max:.4e}")

    # Ensure you write to paths that indicate IQ vs Amp in filename to avoid mixups
    suffix = "IQ" if use_I_Q else "Amp"
    train_norm_path = os.path.join(memmap_dir, f"train_norm_{final_train_chunks_raw.shape[0]}x{window_size}_{suffix}_{np.dtype(dtype).name}.dat")
    val_norm_path   = os.path.join(memmap_dir, f"val_norm_{final_val_chunks_raw.shape[0]}x{window_size}_{suffix}_{np.dtype(dtype).name}.dat")
    test_norm_path  = os.path.join(memmap_dir, f"test_norm_{final_test_chunks_raw.shape[0]}x{window_size}_{suffix}_{np.dtype(dtype).name}.dat")

    # ... (Call blockwise_normalize_to_path) ...
    train_norm_path = blockwise_normalize_to_path(final_train_chunks_raw, train_norm_path, normalization_type, norm_params, dtype=dtype, shape=final_train_chunks_raw.shape)
    val_norm_path   = blockwise_normalize_to_path(final_val_chunks_raw,   val_norm_path,   normalization_type, norm_params, dtype=dtype, shape=final_val_chunks_raw.shape)
    test_norm_path  = blockwise_normalize_to_path(final_test_chunks_raw,  test_norm_path,  normalization_type, norm_params, dtype=dtype, shape=final_test_chunks_raw.shape)

    # --- A1: Normalize the clean injected windows --- #
    # IMPORTANT: the clean component is only SCALED, never mean-shifted.
    # In z-score space the outside-signal baseline must remain exactly zero,
    # so clean_norm = clean_raw / std (NOT (clean_raw - mean) / std).
    clean_norm_paths = {'train': None, 'val': None, 'test': None}
    if save_clean_signals:
        if normalization_type == 'zscore':
            clean_norm_params = {'mean_value': 0.0, 'std_dev_value': norm_params['std_dev_value']}
        elif normalization_type == 'min_max':
            # Pure scaling by the data range (zero baseline preserved).
            clean_norm_params = {
                'min_value': 0.0,
                'max_value': norm_params['max_value'] - norm_params['min_value'],
            }
        else:
            raise ValueError(f"Unsupported normalization_type '{normalization_type}' for clean signals")
        for split_name, n_chunks in (
            ('train', final_train_chunks_raw.shape[0]),
            ('val', final_val_chunks_raw.shape[0]),
            ('test', final_test_chunks_raw.shape[0]),
        ):
            clean_norm_path = os.path.join(
                memmap_dir,
                f"{split_name}_clean_norm_{n_chunks}x{window_size}_{suffix}_{np.dtype(dtype).name}.dat",
            )
            clean_norm_paths[split_name] = blockwise_normalize_to_path(
                clean_arrays[split_name], clean_norm_path, normalization_type,
                clean_norm_params, dtype=dtype, shape=clean_arrays[split_name].shape,
            )

    # --- Info dict with every artefact path (clean signals + metadata) --- #
    info = {
        "train_norm_path": train_norm_path,
        "val_norm_path": val_norm_path,
        "test_norm_path": test_norm_path,
        "train_lbl_path": train_lbl_path,
        "val_lbl_path": val_lbl_path,
        "test_lbl_path": test_lbl_path,
        "train_snr_path": train_snr_path,
        "val_snr_path": val_snr_path,
        "test_snr_path": test_snr_path,
        "train_clean_raw_path": clean_raw_paths['train'],
        "val_clean_raw_path": clean_raw_paths['val'],
        "test_clean_raw_path": clean_raw_paths['test'],
        "train_clean_norm_path": clean_norm_paths['train'],
        "val_clean_norm_path": clean_norm_paths['val'],
        "test_clean_norm_path": clean_norm_paths['test'],
        "event_metadata_path": event_metadata_path if save_metadata else None,
        "window_metadata_path": window_metadata_path if save_metadata else None,
        "train_shape": tuple(final_train_chunks_raw.shape),
        "val_shape": tuple(final_val_chunks_raw.shape),
        "test_shape": tuple(final_test_chunks_raw.shape),
        "response_mode": response_mode,
        "window_size": window_size,
        "step_size": step_size,
        "normalization_type": normalization_type,
        "norm_params": norm_params,
        "fs_val": fs_val,
        "num_channels": num_channels,
        "dtype": np.dtype(dtype).name,
        "no_overlap_injections": bool(no_overlap_injections),
        "no_overlap_margin_samples": int(no_overlap_margin),
        "skipped_no_overlap_counts": dict(skipped_no_overlap_counts),
        "random_seed": random_seed,
        "reject_unrepresentable_injections": reject_unrepresentable_injections,
    }

    # --- TF Datasets --- #
    if return_tf_datasets:
        # Shape depends on mode
        train_shape = final_train_chunks_raw.shape
        val_shape = final_val_chunks_raw.shape
        test_shape = final_test_chunks_raw.shape

        if include_clean_in_datasets and not save_clean_signals:
            raise ValueError(
                "include_clean_in_datasets=True requires save_clean_signals=True "
                "and inject_signals=True so clean_signal_windows exist."
            )

        train_ds = make_train_dataset_from_memmap(
            train_norm_path,
            train_lbl_path,
            train_shape,
            dtype,
            batch_size=tf_batch_size,
            channels=num_channels,
            clean_path=clean_norm_paths['train'] if include_clean_in_datasets else None,
        )

        val_ds = make_eval_dataset_from_memmap(
            val_norm_path,
            val_lbl_path,
            val_shape,
            dtype,
            batch_size=tf_batch_size,
            channels=num_channels,
            clean_path=clean_norm_paths['val'] if include_clean_in_datasets else None,
        )

        test_ds = make_eval_dataset_from_memmap(
            test_norm_path,
            test_lbl_path,
            test_shape,
            dtype,
            batch_size=tf_batch_size,
            channels=num_channels,
            clean_path=clean_norm_paths['test'] if include_clean_in_datasets else None,
        )

    if return_tf_datasets:
        if return_info:
            return train_ds, val_ds, test_ds, info
        return train_ds, val_ds, test_ds
    else:
        if return_info:
            return info
        # Return all relevant paths and arrays, including SNR memmap paths
        return (
            final_train_chunks_raw, final_train_labels, final_val_chunks_raw, final_val_labels, final_test_chunks_raw, final_test_labels,
            train_norm_path, val_norm_path, test_norm_path,
            train_lbl_path, val_lbl_path, test_lbl_path,
            train_snr_path, val_snr_path, test_snr_path,
            final_train_chunks_raw.shape, final_val_chunks_raw.shape, final_test_chunks_raw.shape,
            dtype, 0, 0, 0, 0, fs_val
        )
