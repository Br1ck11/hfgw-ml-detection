import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import math
import os

from data_pre_processing.tiq_data_loader import load_tiq_data_segment
from data_pre_processing.window_data import window_segment
from data_pre_processing.stats import stream_welford_stats, stream_min_max, blockwise_normalize_to_path

from data_pre_processing.chirp_BW_conv_signal_generation import (
    warm_trimmed_cache,
    get_trimmed_waveform,
    clear_trimmed_cache
)

# --- Streaming TF dataset from memmaps --- #

def make_tf_dataset_from_memmap(data_path, labels_path, shape, dtype, batch_size=512, channels=1, shuffle=True, seed=42, repeat=False, prefetch=True):
    import numpy as _np
    import tensorflow as _tf

    def gen():
        X = _np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
        Y = _np.memmap(labels_path, mode="r", dtype=_np.bool_, shape=(shape[0],)) if labels_path is not None else None
        n = shape[0]
        idx = _np.arange(n)
        rng = _np.random.default_rng(seed)
        
        def one_epoch():
            if shuffle:
                rng.shuffle(idx)
            for i in range(0, n, batch_size):
                sel = idx[i:i+batch_size]
                xb = X[sel]
                xb = xb[..., None] if channels == 1 else xb.reshape(xb.shape[0], xb.shape[1], channels)
                if Y is None:
                    yield (xb.astype(dtype),)
                else:
                    yb = Y[sel]
                    yield (xb.astype(dtype), yb.astype(_np.bool_))
                    
        if repeat:
            while True:
                yield from one_epoch()
        else:
            yield from one_epoch()

    output_x = _tf.TensorSpec(shape=(None, shape[1], channels), dtype=_tf.as_dtype(_np.dtype(dtype)))
    if labels_path is None:
        ds = _tf.data.Dataset.from_generator(gen, output_signature=(output_x,))
    else:
        output_y = _tf.TensorSpec(shape=(None,), dtype=_tf.bool)
        ds = _tf.data.Dataset.from_generator(gen, output_signature=(output_x, output_y))
        
    def to_tensor(*batch):
        if len(batch) == 1:
            (x,) = batch
            return _tf.convert_to_tensor(x)
        x, y = batch
        return _tf.convert_to_tensor(x), _tf.convert_to_tensor(y)
        
    ds = ds.map(to_tensor, num_parallel_calls=_tf.data.AUTOTUNE)
    
    if prefetch:
        ds = ds.prefetch(_tf.data.AUTOTUNE)

    # If we are not repeating infinitely, we can calculate and set the exact size.
    if not repeat:
        num_steps = int(_np.ceil(shape[0] / batch_size))
        ds = ds.apply(_tf.data.experimental.assert_cardinality(num_steps))
    # ----------------------------------------

    return ds
    

def pre_processing_with_memmap(
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
    # ----------------------------
    f0_gw=5.0e9, Gamma_gw=100e3, N_gw=32768, M_solar=1.988e30,
    # Cavity response (A5): "real_lorentzian" (legacy) or "complex_breit_wigner".
    # Added here too so efficiency/FP evaluation (which imports this backup
    # module) can match the simulator setting a model was trained with.
    response_mode='real_lorentzian',
    memmap_dir='./memmaps',
    return_tf_datasets: bool = True,
    tf_batch_size: int = 512,
    tf_shuffle: bool = True,
    tf_repeat: bool = False
):
    # --- 1. MODE VALIDATION ---
    if not use_amps and not use_I_Q:
        raise ValueError("Configuration Error: You must select a mode of operation. Set either 'use_amps=True' or 'use_I_Q=True'.")
    
    if use_amps and use_I_Q:
        raise ValueError("Configuration Error: Ambiguous mode. Please set ONLY ONE to True (use_amps OR use_I_Q), not both.")
        
    # Determine channels for final shape
    # If Amps: (N, Window) -> TF expands to (N, Window, 1)
    # If I/Q:  (N, Window, 2) -> TF keeps (N, Window, 2)
    num_channels = 2 if use_I_Q else 1
    
    print(f"--- Preprocessing Mode: {'I/Q (2 Channels)' if use_I_Q else 'Amplitude (1 Channel)'} ---")

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
            memmap_dir=memmap_dir,
            return_tf_datasets=False
        )

        # Load the computed stats from disk (written at end of function)
        global_mean = np.load(os.path.join(memmap_dir, "global_mean.npy"))
        global_std  = np.load(os.path.join(memmap_dir, "global_std.npy"))

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
            print(f"--- Warming Signal Cache (SNR Mode) ---")
            warm_trimmed_cache(m_PBH_injection_list, [1.0], f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)
        else:
            print(f"--- Warming Signal Cache (Raw Amp Mode) ---")
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

    final_train_chunks_raw = np.memmap(train_path, mode='w+', dtype=dtype, shape=memmap_shape_train)
    final_val_chunks_raw   = np.memmap(val_path,   mode='w+', dtype=dtype, shape=memmap_shape_val)
    final_test_chunks_raw  = np.memmap(test_path,  mode='w+', dtype=dtype, shape=memmap_shape_test)

    final_train_labels = np.memmap(train_lbl_path, mode='w+', dtype=np.bool_, shape=(total_train_chunks,))
    final_val_labels   = np.memmap(val_lbl_path,   mode='w+', dtype=np.bool_, shape=(total_val_chunks,))
    final_test_labels  = np.memmap(test_lbl_path,  mode='w+', dtype=np.bool_, shape=(total_test_chunks,))

    # Initialize
    final_train_chunks_raw[:] = 0; final_val_chunks_raw[:] = 0; final_test_chunks_raw[:] = 0
    final_train_labels[:] = False; final_val_labels[:] = False; final_test_labels[:] = False
    
    train_idx_counter = 0; val_idx_counter = 0; test_idx_counter = 0

    # --- PASS 2: Processing files --- #
    print("\n--- Pass 2: Processing files and filling arrays ---")
    rng = np.random.default_rng()

    for i, suffix_string in enumerate(filepath_suffixes):
        # ... (Load Data logic remains same) ...
        current_filepath = filepath_template.format(suffix_string)
        file_label = f"File {i+1} ({current_filepath})"
        print(f"\nProcessing {file_label}...")
        iq_channel_data, fs_val = load_tiq_data_segment(current_filepath, offset, num_samples_to_read_per_file)
        if iq_channel_data is None: continue
            
        file_data_length = len(iq_channel_data)
        train_len = int(file_data_length * train_ratio)
        val_len = int(file_data_length * val_ratio)
        
        segments_to_process = {
            'train': (iq_channel_data[0:train_len], final_train_chunks_raw, final_train_labels, train_idx_counter),
            'val': (iq_channel_data[train_len : train_len + val_len], final_val_chunks_raw, final_val_labels, val_idx_counter),
            'test': (iq_channel_data[train_len + val_len :], final_test_chunks_raw, final_test_labels, test_idx_counter),
        }

        for segment_type, (raw_segment_complex, final_chunks_arr, final_labels_arr, start_idx) in segments_to_process.items():
            if len(raw_segment_complex) < window_size: continue
            segment_length = len(raw_segment_complex)
            has_signal_flags_for_segment = np.zeros(segment_length, dtype=bool)
            
            # --- SNR Calc --- #
            current_noise_std = 1.0
            if snr_based_injection and inject_signals:
                 if custom_noise_std is not None:
                    current_noise_std = float(custom_noise_std)
                    print(f"Using custom std. deviation of {current_noise_std:.3e}") # DEBUG to see if correct value is used for the later coming SNR calculation
                 else:
                    current_noise_std = np.std(np.abs(raw_segment_complex))
                    print(f"Using std. deviation of currently loaded data file of {current_noise_std:.3e}") # DEBUG to see if correct value is used for the later coming SNR calculation

            # --- Injection --- #
            if inject_signals and m_PBH_injection_list and amplitude_spectrum_range:
                current_injection_count = injection_counts.get(segment_type, 0)
                if current_injection_count > 0:
                    for _ in range(current_injection_count):
                        if rng.random() < signal_injection_probability:
                            m_pbh = rng.choice(m_PBH_injection_list)
                            if snr_based_injection:
                                target_snr = rng.choice(amplitude_spectrum_range)
                                waveform = get_trimmed_waveform(m_pbh, 1.0, f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)
                                if waveform.size > 0:
                                    peak = np.max(np.abs(waveform))
                                    if peak > 0: waveform *= (target_snr * current_noise_std / peak) # rescale amplitude to a given SNR value given either a custom 'current_noise_std' or one from the currently loaded training file
                            else:
                                amp_val = rng.choice(amplitude_spectrum_range)
                                waveform = get_trimmed_waveform(m_pbh, amp_val, f0_gw, Gamma_gw, N_gw, M_solar, relative_threshold_factor=1e-3, response_mode=response_mode)
                            
                            sig_len = waveform.size
                            if sig_len > 0 and sig_len <= segment_length:
                                max_start_index = segment_length - sig_len
                                t_start = rng.integers(0, max_start_index + 1)
                                raw_segment_complex[t_start : t_start + sig_len] += waveform
                                has_signal_flags_for_segment[t_start : t_start + sig_len] = True
                                actual_injected_counts[segment_type] += 1

            # --- LOGIC SPLIT: I/Q vs Amplitude --- #
            current_chunks = None
            
            if use_I_Q:
                # 1. Stack Real and Imaginary parts: Shape (Length, 2)
                complex_stacked = np.stack([np.real(raw_segment_complex), np.imag(raw_segment_complex)], axis=-1)
                
                # 2. Windowing for 2D data
                # sliding_window_view on axis 0 gives shape (Num_Windows, 2, Window_Size)
                # We transpose it to (Num_Windows, Window_Size, 2)
                windows_view = sliding_window_view(complex_stacked, window_size, axis=0)[::step_size]
                current_chunks = windows_view.transpose(0, 2, 1)
                
            elif use_amps:
                # 1. Calculate Amplitude: Shape (Length,)
                amplitude_data = np.abs(raw_segment_complex)
                
                # 2. Windowing for 1D data: Shape (Num_Windows, Window_Size)
                current_chunks = window_segment(amplitude_data, window_size, step_size)

            num_new_chunks = len(current_chunks)
            if num_new_chunks == 0: continue

            # --- Labels ---
            if len(has_signal_flags_for_segment) >= window_size:
                chunk_labels_for_segment = sliding_window_view(has_signal_flags_for_segment, window_shape=window_size)[::step_size].any(axis=1)
            else:
                chunk_labels_for_segment = np.zeros(num_new_chunks, dtype=bool)
                
            # --- Fill ---
            end_idx = start_idx + num_new_chunks
            final_chunks_arr[start_idx:end_idx] = current_chunks
            final_labels_arr[start_idx:end_idx] = chunk_labels_for_segment

            if segment_type == 'train': train_idx_counter += num_new_chunks
            elif segment_type == 'val': val_idx_counter += num_new_chunks
            elif segment_type == 'test': test_idx_counter += num_new_chunks

        print(f"Cumulative chunks: Train={train_idx_counter}, Val={val_idx_counter}, Test={test_idx_counter}")

    # --- Cleanup --- #
    if inject_signals:
        clear_trimmed_cache()
        print("\nSIGNAL INJECTION REPORT: ", actual_injected_counts)

    # --- Normalization --- #
    print("\n--- Normalization ---")
    final_train_chunks_raw.flush(); final_val_chunks_raw.flush(); final_test_chunks_raw.flush()
    
    norm_params = {}
    if normalization_type == 'zscore':
        if calculate_stats is True:
            print("Computing GLOBAL training mean/std for normalization...")
            global_mean, global_std = stream_welford_stats(final_train_chunks_raw)
            # Persist GLOBAL stats for reuse (required for SNR-based injection)
            np.save(os.path.join(memmap_dir, "global_mean.npy"), global_mean)
            np.save(os.path.join(memmap_dir, "global_std.npy"),  global_std)
        else:
            global_mean = global_mean_input
            global_std = global_std_input
        
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

    # --- TF Datasets ---
    if return_tf_datasets:
        # Shape depends on mode
        train_shape = final_train_chunks_raw.shape
        val_shape = final_val_chunks_raw.shape
        test_shape = final_test_chunks_raw.shape
        
        train_ds = make_tf_dataset_from_memmap(train_norm_path, train_lbl_path, train_shape, dtype, batch_size=tf_batch_size, channels=num_channels, shuffle=tf_shuffle, repeat=tf_repeat)
        val_ds   = make_tf_dataset_from_memmap(val_norm_path,   val_lbl_path,   val_shape,   dtype, batch_size=tf_batch_size, channels=num_channels, shuffle=False, repeat=False)
        test_ds  = make_tf_dataset_from_memmap(test_norm_path,  test_lbl_path,  test_shape,  dtype, batch_size=tf_batch_size, channels=num_channels, shuffle=False, repeat=False)
        return train_ds, val_ds, test_ds

    return (final_train_chunks_raw, final_train_labels, final_val_chunks_raw, final_val_labels, final_test_chunks_raw, final_test_labels, train_norm_path, val_norm_path, test_norm_path, train_lbl_path, val_lbl_path, test_lbl_path, final_train_chunks_raw.shape, final_val_chunks_raw.shape, final_test_chunks_raw.shape, dtype, 0, 0, 0, 0, fs_val) # Add fs_val
