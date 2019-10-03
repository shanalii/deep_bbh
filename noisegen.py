import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import gwsurrogate as gws 
import gwtools

# generate gaussian noise
def gaussianNoise(samp_rate_hz = 1000000, amp = (1,1), noiseSigma = (0.2,10), dur_us = 8192):

	# total number of samples in the time-series data
	tot_samps = dur_us * samp_rate_hz // 10**6
	print("total samples: " + str(tot_samps))

	# random values within range if not given via arguments
	noiseSigma = np.random.uniform(*noiseSigma)
	amp = np.random.uniform(*amp)
	
	noise = np.random.normal(0, noiseSigma, tot_samps)
	return noise

# generate bowl-shaped shaded noise: in the form Af^(-P1) + Bf^(P2) in the Fourier domain
def shadedNoise(signal, A, p1, B, p2, f0):

	# power law to make bowl shape in the fourier domain
	fft = np.fft.fft(signal)
	f = np.fft.fftfreq(len(fft),.01)
	a = A * (f+f0)**(-p1)
	b = B * (f+f0)**(p2)
	sfft = fft*(a+b)
	fpos=f[0:int(len(f)/2)]
	sfftpos = sfft[0:int(len(f)/2)]
	plt.loglog(fpos, (np.abs(sfftpos)))
	plt.show()

	# inverse fft to get time-series signal
	shadedSig = np.fft.ifft(sfft)
	plt.plot(shadedSig)
	plt.show()
	return shadedSig

shadedNoise(gaussianNoise(), 10, 2, 10, 2, 1)

# take signals from surrogate waveform function
# segment of this noise that is the same length
# add them together

## where do the h5 files come from and what does it change?
# Surrogate data located at http://www.black-holes.org/surrogates/

#this doesn't work:
#spec = gws.EvaluateSurrogate('surrogates/NRSur7dq2.h5', ell_m=[(2,2), (3,3)])
spec = gws.EvaluateSurrogate('surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5', ell_m=[(2,2), (3,3)])
modes, times, hp, hc = spec(q=1.7, ell=[2], m=[2], mode_sum=False, fake_neg_modes=False)
pdb.set_trace()
# Plot the (2,2) mode
gwtools.plot_pretty(times, [hp, hc],fignum=1)
wf = gwtools.amp(hp+1j*hc)
plt.plot(times,wf,'r')
plt.title('The (%i,%i) mode'%(modes[0][0],modes[0][1]))
plt.xlabel('t/M ')
plt.show()

# do they band-pass filter the data before analyzing it?
# after filtering it, just gaussian noise
# do crappy shaded noise, then bandpass filter, then do analysis 