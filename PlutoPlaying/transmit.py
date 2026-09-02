import numpy as np
import adi
import time

sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -30 # Increase to increase tx power, valid range is -90 to 0 dB

N = 10000 # number of samples to transmit at once
t = np.arange(N)/sample_rate

f = 1e5
samples = 0.5*np.exp(2.0j*np.pi*f*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs


sdr.tx_cyclic_buffer = True

while(True):
    sdr.tx(samples) # transmit the batch of samples once
    input("Transmitting, Press Enter to Stop")
    sdr.tx_destroy_buffer() 
    input("Stopped, Press Enter to Start")

print("done")