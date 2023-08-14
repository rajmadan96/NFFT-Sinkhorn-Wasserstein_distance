# NFFT-Sinkhorn (Approximation of Wasserstein distance)
This contribution features an accelerated computation of the Wasserstein transportation distance and the Sinkhorn divergence by employing nonequispaced fast Fourier transforms (NFFT). The method overcomes numerical stability issues of the straightforward convolution of the standard, fast Fourier transform (FFT). Also, the algorithm is faster than the traditional Sinkhorn algorithm.  Further, the proposed method avoids expensive allocations of the characterizing matrices. With this numerical acceleration, the transportation distance is accessible to probability measures out of reach so far. Numerical experiments using synthetic data and real data affirm the computational advantage and supremacy.

### Prerequisites

The "NFFT3.jl" package has be installed. For more details, please refer to  [https://www-user.tu-chemnitz.de](https://www-user.tu-chemnitz.de/~potts/nfft/) and https://github.com/NFFT/NFFT3.jl. 


**NOTE: The "NFFT3.jl" package should be in the same parent directory.**


### Reference

When you are using this code, please cite the paper.

<a id="1">[1]</a> Rajmadan Lakshmanan, Alois Pichler, and  Daniel Potts. (2022). [Nonequispaced Fast Fourier Transform Boost for the Sinkhorn Algorithm](https://epub.oeaw.ac.at/?arp=0x003e223f). 

This paper also comprehensively explains the NFFT-Sinkhorn (Approximation of Wasserstein distance) algorithm.


## Directory structure

| File/Folder   | Purpose                                                                                   |
| ------------- |-------------------------------------------------------------------------------------------|   
| src           | Sinkhorn algorithm from Section 3.3, and NFFT-accelerated  from Section 4.1 of [[1]](#1) |
| Graphs and images        |  Graphs and images of the numerical experiments.               |
| Experiments | Experiment scripts--synthetic data and real data.       |

