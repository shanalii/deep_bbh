from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np 
from torch.utils.data import Dataset
from torch import nn
import torch
import time
import pdb
import NRSur7dq2
import matplotlib.pyplot as plt


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ShadedNoiseDL(BaseDataLoader):
    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, 
        samp_rate_hz=1000., num_samps=5, dur_s=8.192):
        self.data_dir = data_dir
        self.dataset = ShadedNoiseDS(samp_rate_hz=samp_rate_hz, num_samps=num_samps, dur_s=dur_s)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ShadedNoiseDS(Dataset):   

    def __init__(self, samp_rate_hz, num_samps, dur_s):
        self.num_samps = num_samps
        self.dur_s = dur_s
        self.samp_rate_hz = samp_rate_hz
        self.dt = 1./samp_rate_hz

    # return number of samples
    def __len__(self):
        return self.num_samps

    # get the next time-series sample
    def __getitem__(self, idx):

        # .5 chance of only noise
        choice = np.random.choice([True, True, False])
        noise = self.shadedNoise(self.gaussianNoise(self.dur_s, self.samp_rate_hz), 10, 2, 10, 2, 1, self.dt)

        if (choice):
            wf = self.genwf()
            detectedSig = self.injectSig(wf, np.zeros(len(noise)))
            
            # for bandpassing the signal - preprocessing
            #bpSig = self.bandpass(detectedSig)
            #plt.plot(bpSig)
            # plt.plot(detectedSig)
            # plt.show()
            signal = detectedSig + noise
        else:
            signal = noise

        # plt.plot(signal)
        # if (choice): plt.plot(detectedSig)
        # plt.show()
        signal = np.expand_dims(signal, axis=0).astype(np.float32)
        # print(signal)
        return (signal, torch.FloatTensor([0,1]))

    # generate gaussian noise
    def gaussianNoise(self, dur_s, samp_rate_hz, amp=(1,1), noise_sigma=(1.e-10, 2.e-10)):

        # total number of samples in the time-series data
        tot_samps = int(dur_s * samp_rate_hz)
        #print("total samples: " + str(tot_samps))

        # random values within range if not given via arguments
        noise_sigma = np.random.uniform(*noise_sigma)
        amp = np.random.uniform(*amp)
        
        noise = np.random.normal(0, noise_sigma, tot_samps)
        return noise

    # generate bowl-shaped shaded noise: in the form Af^(-P1) + Bf^(P2) in the Fourier domain
    def shadedNoise(self, signal, A, p1, B, p2, f0, dt):

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

    # generate gravitational waveform using NRSur7dq2 surrogate
    def genwf(self, q = 1.7, chiA = .8, thetaA = .1*np.pi, phiA = 0., chiB = .5, thetaB = .7*np.pi, 
        phiB = 0.):
        
        # Surrogate data located at http://www.black-holes.org/surrogates/
        sur = NRSur7dq2.NRSurrogate7dq2("/home/shanali/deep_bbh_libs/NRSur7dq2-1.0.5/NRSur7dq2/NRSur7dq2.h5")
        chiA0 = np.array([chiA*np.sin(thetaA)*np.cos(phiA), chiA*np.sin(thetaA)*np.sin(phiA), chiA*np.cos(thetaA)])
        chiB0 = np.array([chiB*np.sin(thetaB)*np.cos(phiB), chiB*np.sin(thetaB)*np.sin(phiB), chiB*np.cos(thetaB)])

        ## longer time?
        #np.arange(-7., 0.03, self.dt)
        sample_times = np.arange(-1., .03, self.dt)
        #print(sample_times)

        # TODO future: analyze phase info
        ## phi = A(t) e^(i*phi(t))
        # A(t) = sqrt(phi*conj(phi))
        # phi(t) = 1/i log(phi(t)/A(t))
        #h = sur(q, chiA0, chiB0, theta=np.pi/2., phi=0, MTot = 65., distance=1000, t=sample_times)
        ## error here???
        h = sur(q, chiA0, chiB0, theta=np.pi/2., phi=0, MTot = 100., distance=1000, t=sample_times)
        h_plus=np.real(h)
        h_cross=np.imag(h)
        
        # plt.plot(sample_times, h_plus)
        # plt.plot(sample_times, h_cross)
        # plt.ylim(-8.e-22, 8.e-22)
        # plt.show()
        return h_plus

    # takes waveform and noise, place waveform into random position and add them
    def injectSig(self, wf, noise):
        tot_samps = len(noise)
        start_samp_num = np.random.randint(tot_samps - len(wf))
        wfpad = np.pad(wf, (start_samp_num, tot_samps - start_samp_num - len(wf)), 'constant', constant_values=(0,0))
        combined = noise + wfpad
        return combined