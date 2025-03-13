import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

def fft_analysis(file_path, volume_label="", key_peaks=None):
    fs, data = wavfile.read(file_path)
    
    if data.ndim > 1:
        data = data[:, 0]

    data = data / np.max(np.abs(data))

    fft_data = fft(data)
    freqs = np.fft.fftfreq(len(data), 1/fs)

    positive_indices = np.where(freqs >= 0)
    freqs = freqs[positive_indices]
    magnitude = 20 * np.log10(np.abs(fft_data[positive_indices]) + 1e-12)

    valid_indices = magnitude >= 0
    freqs = freqs[valid_indices]
    magnitude = magnitude[valid_indices]

    plt.figure(figsize=(20, 12))
    plt.plot(freqs, magnitude, label="FFT Spectrum")

    # Highlight key peaks
    if key_peaks:
        for peak in key_peaks:
            closest_idx = np.argmin(np.abs(freqs - peak))
            plt.scatter(freqs[closest_idx], magnitude[closest_idx], color='red', s=100, label=f"Peak: {freqs[closest_idx]:.2f} Hz")

    plt.title(f"FFT Spectrum - {volume_label}")
    plt.xticks(np.arange(0, max(freqs), step=1000))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.legend()
    plt.savefig(f"FFT_Spectrum_{volume_label}.png")
    plt.close()

    return fs, freqs, magnitude

def find_nearest_peak(freqs, magnitudes, target_freq, tolerance=5.0):
    valid_indices = np.where((freqs > target_freq - tolerance) & (freqs < target_freq + tolerance))[0]
    if len(valid_indices) == 0:
        print(f"No peaks found near {target_freq} Hz within ±{tolerance} Hz.")
        return target_freq, 0

    peak_index = valid_indices[np.argmax(magnitudes[valid_indices])]
    print(f"Peak found at {freqs[peak_index]:.2f} Hz with magnitude {magnitudes[peak_index]:.5f}")
    return freqs[peak_index], magnitudes[peak_index]

def calculate_ip3(frequencies, magnitudes, f1, f2):
    f_imd1 = 2 * f1 - f2
    f_imd2 = 2 * f2 - f1

    f1_freq, f1_power = find_nearest_peak(frequencies, magnitudes, f1)
    f2_freq, f2_power = find_nearest_peak(frequencies, magnitudes, f2)
    imd1_freq, imd1_power = find_nearest_peak(frequencies, magnitudes, f_imd1)
    imd2_freq, imd2_power = find_nearest_peak(frequencies, magnitudes, f_imd2)

    f1_power_db = 20 * np.log10(f1_power) if f1_power > 0 else -np.inf
    f2_power_db = 20 * np.log10(f2_power) if f2_power > 0 else -np.inf
    imd1_power_db = 20 * np.log10(imd1_power) if imd1_power > 0 else -np.inf
    imd2_power_db = 20 * np.log10(imd2_power) if imd2_power > 0 else -np.inf

    ip3_lower = f1_power_db + (f1_power_db - imd1_power_db) / 2
    ip3_upper = f2_power_db + (f2_power_db - imd2_power_db) / 2

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

file_paths = {
    "10% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_10.wav",
    "30% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_30.wav",
    "50% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_50.wav",
    "70% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_70.wav",
    "100% Volume": "D:/uni/DSP/DSP_IPX_Project/Recording_100.wav"
}

for volume, path in file_paths.items():
    print(f"Analyzing {volume} recording...")

    fs, frequencies, magnitudes = fft_analysis(path, volume_label=volume)

    sorted_indices = np.argsort(magnitudes)[-2:]
    detected_frequencies = frequencies[sorted_indices]
    detected_magnitudes = magnitudes[sorted_indices]
    detected_frequencies = sorted(detected_frequencies)

    f1_detected, f2_detected = detected_frequencies
    print(f"Detected f1: {f1_detected:.2f} Hz, f2: {f2_detected:.2f} Hz")

    ip3_results = calculate_ip3(frequencies, magnitudes, f1_detected, f2_detected)
    # imd1_detected = 2 * f1_detected - f2_detected
    # imd2_detected = 2 * f2_detected - f1_detected
    
    key_peaks = [f1_detected, f2_detected]
    fft_analysis(path, volume_label=volume, key_peaks=key_peaks)

    print(f"\nIP3 Calculation Results for {volume}:")
    for key, value in ip3_results.items():
        if "Frequency" in key:
            print(f"{key}: {value:.2f} Hz")
        else:
            print(f"{key}: {value:.2f} dB")

    print("-" * 50)
