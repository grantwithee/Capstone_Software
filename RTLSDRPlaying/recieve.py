import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rtlsdr import RtlSdr

# --- Configuration ---
CENTER_FREQ = 915e6  # 915 MHz
SAMPLE_RATE = 2.4e6    # 2.4 MHz sample rate
GAIN = 'auto'          # SDR Gain ('auto' or numeric value like 40)
FFT_SIZE = 1024        # Number of frequency bins
WATERFALL_DEPTH = 100  # Number of time rows to display

# --- SDR Setup ---
sdr = RtlSdr()
try:
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = GAIN
except Exception as e:
    print(f"Error configuring SDR: {e}")
    sdr.close()
    exit()

# Calculate actual frequency bins for the X-axis
freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1/SAMPLE_RATE)) + CENTER_FREQ
freq_mhz = freqs / 1e6

# Initialize empty waterfall buffer (Time x Frequency)
waterfall_data = np.full((WATERFALL_DEPTH, FFT_SIZE), -60.0)

# --- Plot Setup ---
fig, (ax_spec, ax_water) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("RTL-SDR Real-Time Spectrum & Waterfall", fontsize=14, fontweight='bold')

# Top Plot: Live Line Spectrum
line, = ax_spec.plot(freq_mhz, waterfall_data[0], color='#00ff00', lw=1)
ax_spec.set_ylabel("Power (dB)")
ax_spec.set_ylim(-60, 10)
ax_spec.grid(True, linestyle='--', alpha=0.5)

# Bottom Plot: Waterfall Image
# origin='upper' means new data enters at the top and scrolls down
im = ax_water.imshow(waterfall_data, aspect='auto', cmap='viridis', 
                     extent=[freq_mhz[0], freq_mhz[-1], WATERFALL_DEPTH, 0],
                     vmin=-50, vmax=5)
ax_water.set_xlabel("Frequency (MHz)")
ax_water.set_ylabel("Time Steps (History)")

plt.tight_layout()

# --- Animation Update Loop ---
def update(frame):
    global waterfall_data
    
    # 1. Read raw IQ samples from SDR
    # Reading roughly 2x FFT_SIZE ensures we have enough data for a clean window
    samples = sdr.read_samples(FFT_SIZE * 2)
    
    # 2. Take the first FFT_SIZE samples and apply a Hanning window
    windowed_samples = samples[:FFT_SIZE] * np.hanning(FFT_SIZE)
    
    # 3. Compute FFT and convert to power magnitude
    fft_output = np.fft.fft(windowed_samples)
    fft_shifted = np.fft.fftshift(fft_output)
    psd = (np.abs(fft_shifted) / FFT_SIZE) ** 2
    
    # 4. Convert to Decibels (dB), adding a tiny offset to avoid log(0)
    db_data = 10 * np.log10(psd + 1e-12)
    
    # 5. Roll waterfall matrix up and insert new row at index 0
    waterfall_data = np.roll(waterfall_data, shift=1, axis=0)
    waterfall_data[0, :] = db_data
    
    # 6. Update plots
    line.set_ydata(db_data)
    im.set_array(waterfall_data)
    
    return line, im

# Run the animation
try:
    ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
    plt.show()
finally:
    # Always cleanly close the SDR handle when exiting
    print("Closing SDR connection...")
    sdr.close()