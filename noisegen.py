import numpy as np 
import matplotlib.pyplot as plt 
import pdb
#import gwsurrogate as gws 
#import gwtools
import NRSur7dq2

# LIGO sample rate: 16384
sample_rate_hz = 1000.
dt = 1./sample_rate_hz

# generate gaussian noise
def gaussianNoise(amp = (1,1), noiseSigma = (0.2,1), dur_s = 8.192):

	# total number of samples in the time-series data
	tot_samps = int(dur_s * sample_rate_hz)
	print("total samples: " + str(tot_samps))

	# random values within range if not given via arguments
	noiseSigma = np.random.uniform(*noiseSigma)
	amp = np.random.uniform(*amp)
	
	noise = np.random.normal(0, noiseSigma, tot_samps)
	return noise

# generate bowl-shaped shaded noise: in the form Af^(-P1) + Bf^(P2) in the Fourier domain
def shadedNoise(signal, A, p1, B, p2, f0):

	# power law to make bowl shape in the fourier domain
	fft = np.fft.rfft(signal)
	f = np.fft.fftfreq(len(fft),dt)
	a = A * (f+f0)**(-p1)
	b = B * (f+f0)**(p2)
	sfft = fft*(a+b)
	fpos=f[0:int(len(f)/2)]
	sfftpos = sfft[0:int(len(f)/2)]
	# plt.loglog(fpos, (np.abs(sfftpos)))
	# plt.show()

	# inverse fft to get time-series signal
	shadedSig = np.fft.irfft(sfft)
	# plt.plot(shadedSig)
	# plt.show()
	return shadedSig

# generate gravitational waveform using NRSur7dq2
def genwf(q = 1.7, chiA = .8, thetaA = .1*np.pi, phiA = 0., chiB = .5, thetaB = .7*np.pi, 
	phiB = 0.):
	# Surrogate data located at http://www.black-holes.org/surrogates/
	sur = NRSur7dq2.NRSurrogate7dq2("/home/shanali/deep_bbh_libs/NRSur7dq2-1.0.5/NRSur7dq2/NRSur7dq2.h5")

	chiA0 = np.array([chiA*np.sin(thetaA)*np.cos(phiA), chiA*np.sin(thetaA)*np.sin(phiA), chiA*np.cos(thetaA)])
	chiB0 = np.array([chiB*np.sin(thetaB)*np.cos(phiB), chiB*np.sin(thetaB)*np.sin(phiB), chiB*np.cos(thetaB)])

	sample_times = np.arange(-7., 0.03, dt)

	h = sur(q, chiA0, chiB0, theta=np.pi/2., phi=0, MTot = 65., distance=1000, t=sample_times)
	h_plus=np.real(h)
	h_cross=np.imag(h)
	
	# plt.plot(sample_times, h_plus)
	# plt.plot(sample_times, h_cross)
	# plt.ylim(-8.e-22, 8.e-22)
	# plt.show()

	return h_plus

# takes waveform and noise, place waveform into given position and add them
def injectSig(wf, noise, start_time_s = 0):
	begin = int(start_time_s * sample_rate_hz)
	wfpad = np.pad(wf, (begin, len(noise)-begin-len(wf)), 'constant', constant_values=(0,0))
	combined = noise + wfpad
	return combined

# performs bandpass filter on signal given starting freq, stopping freq, and filter slope/width
## something wrong with startf and stopf
def bandpass(signal, startf=10, stopf=30, w=1):
	# in fourier domain, multiply by sigmoids with low cutoff and high cutoff
	fft = np.fft.rfft(signal)
	f = np.fft.rfftfreq(len(signal),dt)
	pdb.set_trace()
	a = 1/(1+np.exp(-(1/w)*(f-startf)))
	b = 1/(1+np.exp(-(1/w)*(-f+stopf)))
	sfft = fft*(a*b)
	fpos=f[0:int(len(f)/2)]
	filtsig = np.fft.irfft(sfft)
	plt.plot(f,a*b)
	plt.show()
	pdb.set_trace()
	# plt.plot(filtsig)
	# plt.show()
	return filtsig

# do things:
noise = shadedNoise(gaussianNoise(), 10, 2, 10, 2, 1)
wf = genwf()
detectedSig = injectSig(wf, np.zeros(len(noise)))
bpSig = bandpass(detectedSig)
plt.plot(detectedSig)
plt.plot(bpSig)
plt.show()
# does the noise have proper amplitude?

# todo: readme (get rob to proofread)
# 

# nyquist = sr/2, that is the highest frequency we can get via fft
# aliasing is not occuring based on what we see in fft (not growing with frequency)
# max frequency: 1/(fmax) = Mtot*G/c^3

#this doesn't work:
#spec = gws.EvaluateSurrogate('surrogates/NRSur7dq2.h5', ell_m=[(2,2), (3,3)])
# spec = gws.EvaluateSurrogate('surrogates/SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0.h5', ell_m=[(2,2), (3,3)])
# modes, times, hp, hc = spec(q=1.7, ell=[2], m=[2], mode_sum=False, fake_neg_modes=False)
# pdb.set_trace()
# # Plot the (2,2) mode
# gwtools.plot_pretty(times, [hp, hc],fignum=1)
# wf = gwtools.amp(hp+1j*hc)
# plt.plot(times,wf,'r')
# plt.title('The (%i,%i) mode'%(modes[0][0],modes[0][1]))
# plt.xlabel('t/M ')
# plt.show()





# do they band-pass filter the data before analyzing it?
# after filtering it, just gaussian noise
# do crappy shaded noise, add wf, then bandpass filter (10Hz to 10kHz), then do analysis 

# 2 bandpass filters, one for positive and one for negative freqs
# or do rfft to only consider only real valued signals