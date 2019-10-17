# Deep Learning for Binary Black Hole Detection and Parameter Estimation with Numerical Relativity Waveform Surrogates

## Introduction ##

A deep learning project that aims to improve upon the accuracy and efficiency of LIGO's existing binary black hole detection algorithms, which conventionally utilize matched-filtering searches. The goal of this project is to construct, train, and optimize deep neural networks that analyze simulations of LIGO's time-series data, correctly identifies the presence of gravitational waves from binary black hole systems, and estimates the multi-dimensional parameters of the binary system. The initial deep learning pipeline replicates the architecture of the "deep filter" neural networks presented by George and Huerta in [3]. The simulated data is produced using a combination of numerical relativity waveform surrogate models [4] and other general relativity simulation methods. By using a variety of waveform simulation methods, this project also aims to improve upon the efficiency of existing attempts to detect and parameter estimate binary systems via deep learning, as well as expanding the limited parameter space that current methods are able to consider.


## Resources ##

The following resources have been referenced or used in this project:

1. [PyTorch Template by victoresque](https://github.com/victoresque/pytorch-template)

2. [LIGO's Gravitational-Wave Open Science Center](https://www.gw-openscience.org/about/)

3. George, Daniel, and E. A. Huerta. “Deep Neural Networks to Enable Real-Time Multimessenger Astrophysics.” Physical Review D 97.4 (2018): n. pag. Crossref. Web.

4. Blackman, Jonathan et al. “Numerical Relativity Waveform Surrogate Model for Generically Precessing Binary Black Hole Mergers.” Physical Review D 96.2 (2017): n. pag. Crossref. Web.