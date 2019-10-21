from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np 
from torch.utils.data import Dataset
from torch import nn
import torch
import time
import pdb
import NRSur7dq2


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
        samp_rate_hz=1000., num_samps=100, dur_s=8.192):
        self.data_dir = data_dir
        self.dataset = ShadedNoiseDS(samp_rate_hz=1000., num_samps=num_samps, dur_s=dur_us)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ShadedNoiseDS(Dataset):

    def __init__(self, samp_rate_hz=1000., num_samps=100, dur_s=8.192):
        self.num_samps = num_samps
        self.noise_sigma = noise_sigma
        self.samp_rate_hz = samp_rate_hz
        self.dt = 1./sample_rate_hz

    # return number of samples
    def __len__(self):
        return self.num_samps

    # generate gaussian noise
    def gaussianNoise(amp = (1,1), noise_sigma=(.2, .6)):

        # total number of samples in the time-series data
        tot_samps = int(self.dur_s * self.samp_rate_hz)
        print("total samples: " + str(tot_samps))

        # random values within range if not given via arguments
        noise_sigma = np.random.uniform(*noise_sigma)
        amp = np.random.uniform(*amp)
        
        noise = np.random.normal(0, noise_sigma, tot_samps)
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

        sample_times = np.arange(-7., 0.03, self.dt)

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
        begin = int(start_time_s * self.sample_rate_hz)
        wfpad = np.pad(wf, (begin, len(noise)-begin-len(wf)), 'constant', constant_values=(0,0))
        combined = noise + wfpad
        return combined

    # get the next time-series sample
    def __getitem__(self, idx):
        # .5 chance of only noise
        choice = True
        #np.random.uniform.choice([True, False])
        noise = shadedNoise(gaussianNoise(), 10, 2, 10, 2, 1)

        if (choice):
            wf = genwf()
            detectedSig = injectSig(wf, np.zeros(len(noise)))
            bpSig = bandpass(detectedSig)
            # plt.plot(detectedSig)
            # plt.plot(bpSig)
            # plt.show()
            return bpSig
        else:
            return noise