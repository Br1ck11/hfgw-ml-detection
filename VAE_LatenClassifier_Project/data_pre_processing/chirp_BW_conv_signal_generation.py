import numpy as np
import matplotlib.pyplot as plt
import math
from functools import lru_cache
import numpy as _np
try:
    import scienceplots
    plt.style.use('science')
except:
    print("Scienceplots could not be loaded.")

# Supported cavity response modes.
#   "real_lorentzian"     : existing/legacy behavior — real, normalized
#                            Breit-Wigner (Lorentzian) amplitude response.
#   "complex_breit_wigner": complex single-pole response
#                            H(f) = 1 / ((f - f0) + i*Gamma/2), normalized
#                            to max(|H|) = 1, which also imprints the cavity
#                            phase on the chirp.
VALID_RESPONSE_MODES = ("real_lorentzian", "complex_breit_wigner")


def _validate_response_mode(response_mode):
    if response_mode not in VALID_RESPONSE_MODES:
        raise ValueError(
            f"Unknown response_mode='{response_mode}'. "
            f"Choose one of {VALID_RESPONSE_MODES}."
        )
    return response_mode


def _generate_gw_signal_time_domain_nocache(m_PBH, f0, Gamma, N, M_solar, amplitude_spectrum_base_value,
                                            response_mode="real_lorentzian"):
    """
    Generates a complex time-domain gravitational wave signal based on PBH mass
    and cavity parameters, applying a chirp and spectral shaping.

    Args:
        m_PBH (float): Mass of the Primordial Black Hole in solar masses.
        f0 (float): Resonant frequency of the cavity in Hz.
        Gamma (float): FWHM (bandwidth) of the cavity in Hz.
        N (int): Number of frequency points for the FFT grid. This determines the signal's
                 time-domain duration/resolution.
        M_solar (float): Solar mass in kg.
        amplitude_spectrum_base_value (float): value of chirp amplitude spectrum
        response_mode (str): "real_lorentzian" (default, legacy behavior) or
                 "complex_breit_wigner" (complex cavity response with phase).

    Returns:
        np.ndarray: A complex 1D NumPy array representing the time-domain GW signal.
    """
    _validate_response_mode(response_mode)
    f_span = 20 * Gamma
    f_array = np.linspace(f0 - f_span / 2, f0 + f_span / 2, N)
    f_start = f_array[0]
    #print(f"f_start: {f_start}")

    # Calculate frequency change / chirp rate k
    k = 4.62e11 * (m_PBH / (1e-9 * M_solar))**(5/3) * (f_start / 1e9)**(11/3)

    # Calculate phase spectrum (chirp) under SPA (Stationary-Phase-Approximation)
    phase_spectrum = -(np.pi / k) * (f_array - f_start)**2

    amplitude_spectrum = amplitude_spectrum_base_value
    C_f = amplitude_spectrum * np.exp(1j * phase_spectrum) # Chirp in frequency domain

    # Multiply with cavity response and chirp
    if response_mode == "complex_breit_wigner":
        # Complex single-pole cavity response (carries amplitude AND phase)
        delta_f = f_array - f0
        H = 1.0 / (delta_f + 1j * Gamma / 2)
        H = H / np.max(np.abs(H))
        Output_f = H * C_f
    else:
        # Legacy: real, normed Breit-Wigner distribution s.t. AUC = 1
        B_f = 1 / (2 * np.pi) * Gamma / ((f_array - f0)**2 + (Gamma/2)**2)
        Output_f = B_f * C_f

    # Calculate time axis properties for linear phase shift for centering
    df = f_array[1] - f_array[0]
    time_duration_ifft = 1 / df # This is the total time duration represented by the FFT grid
    t_shift = time_duration_ifft / 2 # Shift to center of the window

    # Apply the linear phase shift in frequency domain for time centering in time domain
    phase_shift_linear = np.exp(-1j * 2 * np.pi * f_array * t_shift)
    Output_f_shifted = Output_f * phase_shift_linear

    # Apply ifftshift before iFFT (standard practice for centered FFTs)
    Output_f_ready_for_ifft = np.fft.ifftshift(Output_f_shifted)

    # Perform iFFT to get time-domain signal
    # The output is complex
    gw_signal_complex_time_domain = np.fft.ifft(Output_f_ready_for_ifft)
    
    return gw_signal_complex_time_domain

@lru_cache(maxsize=2048)
def generate_gw_signal_time_domain(m_PBH, f0, Gamma, N, M_solar, amplitude_spectrum_base_value,
                                   response_mode="real_lorentzian"):
    """LRU-cached wrapper around the heavy generator. Returns a COPY to protect cache from mutation."""
    arr = _generate_gw_signal_time_domain_nocache(m_PBH, f0, Gamma, N, M_solar, amplitude_spectrum_base_value,
                                                  response_mode=response_mode)
    arr = _np.asarray(arr)
    return arr.copy()

@lru_cache(maxsize=4096)
def get_trimmed_waveform(m_PBH, amplitude_spectrum_base_value, f0, Gamma, N, M_solar, relative_threshold_factor=1e-3,
                         response_mode="real_lorentzian"):
    """Return a TRIMMED time-domain waveform (cached) using a relative amplitude threshold.
    Trimming rule: keep samples where abs(waveform) > peak * relative_threshold_factor, crop leading/trailing tails.
    """
    w = generate_gw_signal_time_domain(m_PBH, f0, Gamma, N, M_solar, amplitude_spectrum_base_value,
                                       response_mode=response_mode)
    if w.size == 0:
        return _np.empty(0, dtype=w.dtype)
    peak = _np.max(_np.abs(w))
    if not _np.isfinite(peak) or peak <= 0:
        return _np.empty(0, dtype=w.dtype)
    thr = peak * float(relative_threshold_factor)
    active = _np.flatnonzero(_np.abs(w) > thr)
    if active.size == 0:
        return _np.empty(0, dtype=w.dtype)
    trimmed = w[active[0]:active[-1]+1]
    return trimmed.copy()

@lru_cache(maxsize=1)
def cache_info_trimmed():
    """Return cache stats for the trimmed-waveform cache (hits, misses, maxsize, currsize)."""
    try:
        return get_trimmed_waveform.cache_info()
    except Exception:
        return None

@lru_cache(maxsize=1)
def clear_trimmed_cache():
    """Clear the LRU cache for trimmed waveforms."""
    try:
        get_trimmed_waveform.cache_clear()
    except Exception:
        pass


def warm_trimmed_cache(m_PBH_list, amplitude_list, f0, Gamma, N, M_solar, relative_threshold_factor=1e-3,
                       response_mode="real_lorentzian"):
    """Precompute and cache TRIMMED waveforms for all combinations of masses × amplitudes."""
    if not m_PBH_list or not amplitude_list:
        return
    for m in m_PBH_list:
        for a in amplitude_list:
            _ = get_trimmed_waveform(m, a, f0, Gamma, N, M_solar, relative_threshold_factor,
                                     response_mode=response_mode)

# ---------------- Cache utilities ----------------

def clear_signal_cache():
    """Clear the LRU cache of generated waveforms."""
    try:
        generate_gw_signal_time_domain.cache_clear()
    except Exception:
        pass

def cache_info_signal():
    """Return cache stats (hits, misses, maxsize, currsize), or None if unsupported."""
    try:
        return generate_gw_signal_time_domain.cache_info()
    except Exception:
        return None

def warm_signal_cache(m_PBH_list, amplitude_list, f0, Gamma, N, M_solar):
    """Precompute and cache waveforms for all combinations of masses × amplitudes."""
    if not m_PBH_list or not amplitude_list:
        return
    for m in m_PBH_list:
        for a in amplitude_list:
            _ = generate_gw_signal_time_domain(m, f0, Gamma, N, M_solar, a)

def generate_gw_signal_time_domain_batch(param_list):
    """Batch-generate waveforms via the cached API.
    param_list: iterable of dicts with keys (m_PBH, f0, Gamma, N, M_solar, amplitude_spectrum_base_value)
    Returns: list of np.ndarray
    """
    out = []
    for p in param_list:
        out.append(
            generate_gw_signal_time_domain(
                p['m_PBH'], p['f0'], p['Gamma'], p['N'], p['M_solar'], p['amplitude_spectrum_base_value']
            )
        )
    return out

# Example usage (for testing this function separately):
if __name__ == "__main__":
    M_solar_test = 1.988e30
    # Test a few m_PBH values
    m_PBH_test_low = 1e-12 * M_solar_test
    m_PBH_test_high = 1e-10 * M_solar_test
    m_PBH_test_pulse = 1e-6 * M_solar_test # Example for a very high mass to get a pulse

    f0_test = 5.0e9
    Gamma_test = 100e3
    N_test = 32768
    amplitude_spectrum_base_value = 1.0 # For visualization

    # --- Define f_array (and related params) in the global scope for testing --- #
    f_span_test = 20 * Gamma_test # Define f_span_test
    f_array_test = np.linspace(f0_test - f_span_test / 2, f0_test + f_span_test / 2, N_test)

    print(f"Generating signal for m_PBH={m_PBH_test_low:.2e} kg")
    signal_low_m = generate_gw_signal_time_domain(m_PBH_test_low, f0_test, Gamma_test, N_test, M_solar_test, amplitude_spectrum_base_value)
    print(f"Signal length: {len(signal_low_m)}, type: {signal_low_m.dtype}")

    print(f"Generating signal for m_PBH={m_PBH_test_high:.2e} kg")
    signal_high_m = generate_gw_signal_time_domain(m_PBH_test_high, f0_test, Gamma_test, N_test, M_solar_test, amplitude_spectrum_base_value)

    print(f"Generating signal for m_PBH={m_PBH_test_pulse:.2e} kg")
    signal_pulse = generate_gw_signal_time_domain(m_PBH_test_pulse, f0_test, Gamma_test, N_test, M_solar_test, amplitude_spectrum_base_value)

    # Plotting example
    dt_test = (1 / (f_array_test[1] - f_array_test[0])) / N_test
    time_axis_test = np.arange(N_test) * dt_test

    # --- Trimmed waveforms: compute, print lengths, and plot ---
    relative_threshold_factor = 1e-3  # adjust if you want more/less aggressive trimming

    trimmed_low = get_trimmed_waveform(
        m_PBH_test_low, amplitude_spectrum_base_value, f0_test, Gamma_test, N_test, M_solar_test,
        relative_threshold_factor=relative_threshold_factor
    )
    trimmed_high = get_trimmed_waveform(
        m_PBH_test_high, amplitude_spectrum_base_value, f0_test, Gamma_test, N_test, M_solar_test,
        relative_threshold_factor=relative_threshold_factor
    )
    trimmed_pulse = get_trimmed_waveform(
        m_PBH_test_pulse, amplitude_spectrum_base_value, f0_test, Gamma_test, N_test, M_solar_test,
        relative_threshold_factor=relative_threshold_factor
    )

    print(f"Trimmed lengths (relative_threshold_factor={relative_threshold_factor}):")
    print(f"  low m_PBH:   {trimmed_low.size} samples")
    print(f"  high m_PBH:  {trimmed_high.size} samples")
    print(f"  pulse m_PBH: {trimmed_pulse.size} samples")

    def _plot_trimmed(w, title, name):
        if w.size == 0:
            print(f"[trimmed plot skipped] {title}: empty after trimming")
            return
        # Use the same dt as the full-length signal; start time at 0 for the trimmed slice
        t_trim = np.arange(w.size) * dt_test
        plt.figure(figsize=(12, 5))
        plt.plot(t_trim * 1e6, np.real(w), label='Real (trimmed)', color = "green")
        plt.plot(t_trim * 1e6, np.imag(w), '-', label='Imag (trimmed)')
        # plt.title(f"{title} (len={w.size} samples, thr={relative_threshold_factor:g}")
        plt.xlabel("Time (µs)", fontsize=18)
        plt.xticks(size=16)
        plt.ylabel("Amplitude", fontsize=18)
        plt.yticks(size=16)
        plt.legend(loc='best', fontsize=18)
        plt.grid(True)
        plt.savefig(f'trimmed_signal_{name}.png', dpi=300)
        plt.show()
        plt.close()

    _plot_trimmed(trimmed_low,  f"Trimmed GW Signal (Low m_PBH={m_PBH_test_low:.2e} kg)", "1e_minus_13")
    _plot_trimmed(trimmed_high, f"Trimmed GW Signal (High m_PBH={m_PBH_test_high:.2e} kg)", "1e_minus_12")
    _plot_trimmed(trimmed_pulse, f"Trimmed GW Signal (Pulse m_PBH={m_PBH_test_pulse:.2e} kg)", "1e_minus_6")

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis_test * 1e6, np.real(signal_low_m), label='Real Part (Low m_PBH)')
    plt.plot(time_axis_test * 1e6, np.imag(signal_low_m), label='Imag Part (Low m_PBH)', linestyle='--')
    plt.title(f"GW Signal (Low m_PBH={m_PBH_test_low:.2e})")
    plt.xlabel("Time (µs)")
    plt.ylabel("Arbitrary Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis_test * 1e6, np.real(signal_pulse), label='Real Part (High m_PBH)')
    plt.plot(time_axis_test * 1e6, np.imag(signal_pulse), label='Imag Part (High m_PBH)', linestyle='--')
    plt.title(f"GW Signal (High m_PBH={m_PBH_test_pulse:.2e})")
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()
