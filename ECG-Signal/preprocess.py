import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal as sig
import csv

########  LOAD DATA ################################

f = open("./Our_Device/mohammadreza_salehi/data.txt", "r")
reader = csv.reader(f)
signal1 = []
signal2 = []
for row in reader:
        sp = (row[0].split("\t"))
        signal1.append(float(sp[0]))
        signal2.append(float(sp[1]))


#########   PLOT SIGNALS #############################

plt.figure()
plt.subplot(211)
plt.plot(signal1)
plt.xlim([0, 400])
plt.ylabel("Signal 1")
plt.subplot(212)
plt.plot(signal2)
plt.xlim([0, 400])
plt.ylabel("Signal 2")
plt.show()

######### PLOT FOURIER TRANSFORMS ####################

ECG = signal2

ECG = ECG - np.mean(ECG) # Remove DC component
fs = 125    # Sampling Frequency
T = 1/fs    # Sampling Time
yf = fft(ECG)    # Fourier Transform
xf = np.linspace(0.0, 1.0/(2*T), len(ECG)//2)  # Frequency Axis
plt.figure()
plt.plot(xf, 2./len(ECG) * np.abs(yf[0:len(ECG)//2]))
plt.ylabel("fft(ECG)")
plt.xlabel("Frequency")
plt.show()

