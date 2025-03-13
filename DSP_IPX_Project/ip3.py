import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

# Function to Plot Volume Spectrum
def plot_volume_spectrum(file_path, volume_label=""):
    """
    Plot and save the Volume Spectrum (Time-Domain Amplitude) of an audio file.
    """
    # Read audio file
    fs, data = wavfile.read(file_path)
    
    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalize the audio signal
    data = data / np.max(np.abs(data))
    
    # Generate time axis
    time_axis = np.linspace(0, len(data) / fs, num=len(data))
    
    # Plot the Volume Spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, data, label="Amplitude")
    plt.title(f"Volume Spectrum - {volume_label}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.savefig(f"Volume_Spectrum_{volume_label}.png")  # Save the plot as a PNG file
    plt.close()  # Close the plot to prevent it from displaying

# Function for FFT Analysis
def fft_analysis(file_path, segment_duration=3.0, volume_label=""):
    """
    Perform FFT on an audio file and extract frequencies and magnitudes.
    Save the FFT plot as an image.
    """
    # Read audio file
    fs, data = wavfile.read(file_path)
    
    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalize the audio signal
    data = data / np.max(np.abs(data))
    
    # Use a segment of the signal for FFT
    N = int(fs * segment_duration)
    data_segment = data[:N] if len(data) > N else data

    # Compute FFT
    fft_data = fft(data_segment)
    freqs = np.fft.fftfreq(len(data_segment), 1/fs)
    magnitude = np.abs(fft_data[:len(data_segment)//2])  # Positive frequencies only
    freqs = freqs[:len(data_segment)//2]

    # Save the FFT Spectrum plot
    plt.figure(figsize=(20, 12))
    plt.plot(freqs, 20 * np.log10(magnitude + 1e-12))  # Add a small value to avoid log(0)
    plt.title(f"FFT Spectrum - {volume_label}")
    plt.xticks(np.arange(0, max(freqs), step=1000))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.savefig(f"FFT_Spectrum_{volume_label}.png")  # Save the plot as a PNG file
    plt.close()  # Close the plot to prevent it from displaying

    return fs, freqs, magnitude

# Function to Find Nearest Peaks
def find_nearest_peak(freqs, magnitudes, target_freq, tolerance=5.0):
    """
    Find the closest peak to a target frequency within a given tolerance.
    Returns the frequency and magnitude of the peak.
    """
    # Filter frequencies within the tolerance range
    valid_indices = np.where((freqs > target_freq - tolerance) & (freqs < target_freq + tolerance))[0]
    if len(valid_indices) == 0:
        print(f"No peaks found near {target_freq} Hz within ±{tolerance} Hz.")
        return target_freq, 0  # Return 0 magnitude if no peak is found in range

    # Find the index of the maximum magnitude within the range
    peak_index = valid_indices[np.argmax(magnitudes[valid_indices])]
    print(f"Peak found at {freqs[peak_index]:.2f} Hz with magnitude {magnitudes[peak_index]:.5f}")
    return freqs[peak_index], magnitudes[peak_index]

# Function to Calculate IP3
def calculate_ip3(frequencies, magnitudes, f1, f2):
    """
    Calculate the IP3 (Third-Order Intercept Point).
    """
    # Third-order intermodulation product frequencies
    f_imd1 = 2 * f1 - f2  # Lower product
    f_imd2 = 2 * f2 - f1  # Upper product

    # Find magnitudes for fundamental and intermodulation frequencies
    f1_freq, f1_power = find_nearest_peak(frequencies, magnitudes, f1)
    f2_freq, f2_power = find_nearest_peak(frequencies, magnitudes, f2)
    imd1_freq, imd1_power = find_nearest_peak(frequencies, magnitudes, f_imd1)
    imd2_freq, imd2_power = find_nearest_peak(frequencies, magnitudes, f_imd2)

    # Convert magnitudes to dB
    f1_power_db = 20 * np.log10(f1_power) if f1_power > 0 else -np.inf
    f2_power_db = 20 * np.log10(f2_power) if f2_power > 0 else -np.inf
    imd1_power_db = 20 * np.log10(imd1_power) if imd1_power > 0 else -np.inf
    imd2_power_db = 20 * np.log10(imd2_power) if imd2_power > 0 else -np.inf

    # Calculate IP3
    ip3_lower = f1_power_db + (f1_power_db - imd1_power_db) / 2
    ip3_upper = f2_power_db + (f2_power_db - imd2_power_db) / 2

    # Results
    result = {
        "Fundamental f1 Frequency (Hz)": f1_freq,
        "Fundamental f1 Power (dB)": f1_power_db,
        "Fundamental f2 Frequency (Hz)": f2_freq,
        "Fundamental f2 Power (dB)": f2_power_db,
        "IMD1 Frequency (Hz)": imd1_freq,
        "IMD1 Power (dB)": imd1_power_db,
        "IMD2 Frequency (Hz)": imd2_freq,
        "IMD2 Power (dB)": imd2_power_db,
        "IP3 Lower (2f1 - f2)": ip3_lower,
        "IP3 Upper (2f2 - f1)": ip3_upper,
    }

    return result

# File paths for all recordings
file_paths = {
    "10% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_10.wav",
    "30% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_30.wav",
    "50% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_50.wav",
    "70% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_70.wav",
    "100% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_100.wav"
}

# Analyze each recording
for volume, path in file_paths.items():
    print(f"Analyzing {volume} recording...")

    # Plot and save the Volume Spectrum
    plot_volume_spectrum(path, volume_label=volume)
    
    # Perform FFT Analysis and save the FFT Spectrum
    fs, frequencies, magnitudes = fft_analysis(path, segment_duration=5.0, volume_label=volume)

    # Automatically detect the two strongest peaks in the FFT
    sorted_indices = np.argsort(magnitudes)[-2:]  # Top 2 peaks
    detected_frequencies = frequencies[sorted_indices]
    detected_magnitudes = magnitudes[sorted_indices]

    # Sort detected peaks by frequency
    detected_frequencies = sorted(detected_frequencies)
    detected_magnitudes = sorted(detected_magnitudes)

    f1_detected, f2_detected = detected_frequencies
    print(f"Detected f1: {f1_detected:.2f} Hz, f2: {f2_detected:.2f} Hz")

    # Calculate IP3 using detected f1 and f2
    ip3_results = calculate_ip3(frequencies, magnitudes, f1_detected, f2_detected)

    # Display Results
    print(f"\nIP3 Calculation Results for {volume}:")
    for key, value in ip3_results.items():
        if "Frequency" in key:  # For frequencies, no "dB"
            print(f"{key}: {value:.2f} Hz")
        else:  # For power values, add "dB"
            print(f"{key}: {value:.2f} dB")

    print("-" * 50)  # Separator for clarity
