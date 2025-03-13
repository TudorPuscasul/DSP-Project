import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, windows
import matplotlib.pyplot as plt

def reanalyze_full_fft(file_path, chunk_duration=1.0, tolerance=3.0):
    """
    Reanalyze an audio file by splitting it into chunks, detecting peaks, and cleaning up redundant detections.
    """
    # Read audio file
    fs, data = wavfile.read(file_path)
    
    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]
    
    # Normalize the audio signal
    data = data / np.max(np.abs(data))
    
    # Define chunk size
    chunk_size = int(fs * chunk_duration)
    total_chunks = len(data) // chunk_size

    # Storage for detected peaks
    detected_peaks = []

    for i in range(total_chunks):
        # Extract chunk
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        
        # Apply a window function to reduce spectral leakage
        window = windows.hann(len(chunk))
        chunk_windowed = chunk * window
        
        # Perform FFT
        fft_data = np.fft.fft(chunk_windowed)
        freqs = np.fft.fftfreq(len(chunk_windowed), 1/fs)
        magnitude = np.abs(fft_data)[:len((chunk_windowed)//2)]  # Positive frequencies only
        freqs = freqs[:len(chunk_windowed)//2]

        # Detect peaks using find_peaks
        peak_indices, _ = find_peaks(magnitude, distance=(fs//2000))  # Separate peaks by at least ~50 Hz
        chunk_peak_freqs = freqs[peak_indices]
        chunk_peak_magnitudes = magnitude[peak_indices]

        # Store peaks
        detected_peaks.extend(zip(chunk_peak_freqs, chunk_peak_magnitudes))

    # Deduplicate nearby peaks
    deduplicated_peaks = []
    detected_peaks = sorted(detected_peaks, key=lambda x: x[0])  # Sort by frequency
    for freq, mag in detected_peaks:
        if deduplicated_peaks and abs(deduplicated_peaks[-1][0] - freq) < tolerance:
            # Merge peaks by keeping the one with the highest magnitude
            if mag > deduplicated_peaks[-1][1]:
                deduplicated_peaks[-1] = (freq, mag)
        else:
            deduplicated_peaks.append((freq, mag))

    # Plot FFT with deduplicated peaks
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(magnitude + 1e-12), label="FFT Spectrum")
    for i, (freq, mag) in enumerate(deduplicated_peaks):
        plt.scatter(freq, 20 * np.log10(mag + 1e-12), color='red', label=f"Peak {i+1}: {freq:.2f} Hz")
    plt.title("FFT Spectrum with Deduplicated Peaks (Positive Frequencies Only)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.legend()
    plt.show()

    # Display deduplicated peaks
    print("Deduplicated Peaks:")
    for freq, mag in deduplicated_peaks:
        print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.5f}")

# File path for reanalysis
file_path = "D:/uni/DSP/DSP_IPX_Project/Recording_70.wav"  # Replace with your file path
reanalyze_full_fft(file_path, chunk_duration=1.0)
