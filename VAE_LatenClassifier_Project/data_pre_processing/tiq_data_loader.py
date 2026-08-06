from iqtools import TIQData # Import only what's needed, not '*'
import numpy as np
# No need for matplotlib here, as this module's job is just data loading

def load_tiq_data_segment(filepath, offset, num_samples, desired_channel=0):
    """
    Loads a specific segment of IQ data from a .tiq file.

    Args:
        filepath (str): Path to the .tiq file.
        offset (int): Starting sample offset to read from.
        num_samples (int): Number of samples to read.
        desired_channel (int): Which IQ channel to extract if data is multi-channel.

    Returns:
        tuple: (iq_channel_data, fs), where iq_channel_data is a 1D numpy array
               of complex IQ samples for the selected channel, and fs is the
               sampling frequency.
    """
    try:
        # Load tiq file data using iqtools
        iq_data = TIQData(filepath)
        # print(f"File Center Freq: {iq_data.center} Hz") # Optional: print metadata here
        # print(f"Total Samples in file: {iq_data.nsamples_total}")

        # Read "num_samples" starting at index "offset"
        iq_data.read_samples(num_samples, offset)
        iq_samples = iq_data.data_array

        # Ensure tiq data file is single-channel
        if iq_samples.ndim == 1:
            iq_channel_data = iq_samples
        else:
            raise ValueError(f"Unexpected data shape: {iq_samples.shape}. Can only handle 1D IQ data.")

        return iq_channel_data, iq_data.fs

    except Exception as e:
        print(f"Error loading TIQ data: {e}")
        return None, None # Or raise the error, depending on desired error handling



# This block ensures code inside it only runs when data_loader.py is executed directly
# (e.g., python data_loader.py), not when imported by another script.
if __name__ == "__main__":
    print("Running data_loader.py directly (for testing)...")
    # Example usage for testing this module independently:
    test_filename = '../GravNet/Data/IQDataFile-2024.04.18.19.22.56.276.tiq'
    test_offset = 0
    test_num_samples = 100000 # Read a reasonable amount for testing

    data, sample_rate = load_tiq_data_segment(test_filename, test_offset, test_num_samples)

    if data is not None:
        print(f"Loaded {len(data)} samples at {sample_rate} Hz.")
        # You could add some basic plotting here for testing,
        # but typically data loaders focus on just loading.
    else:
        print("Failed to load test data.")
