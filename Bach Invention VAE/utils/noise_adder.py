# Add noise to corrupt the latent vector
import numpy as np

class NoiseAdder:
    def __init__(self, latent_size):
        self.latent_size = latent_size
    
    def get_bit_allocation(self, z_sample, rate):
        z_mu = np.mean(z_sample, axis = 0)
        z_sigma = np.var(z_sample, axis = 0)
        # z_sigma = np.mean(z_var ** 2, axis = 0)
        bit_alloc = self.bit_allocate(z_sigma, rate)

        chan_mumul = (1-np.power(2,-2*bit_alloc))
        chan_sig2mul = np.power(2,-4*bit_alloc)*(np.power(2,2*bit_alloc)-1)
        chan_eps = np.random.randn(*np.shape(z_sample))

        #Using the actual mean and variance of the encoder Z:
        z_quant = np.matmul(z_mu, np.diag(np.power(2,-2*bit_alloc))) + \
        np.matmul(z_sample,np.diag(chan_mumul)) + \
        np.matmul(np.matmul(chan_eps,np.diag(np.sqrt(chan_sig2mul))),np.diag(np.sqrt(z_sigma)))
        return z_mu, z_sigma, z_sample, z_quant


    def bit_allocate(self, sigma2, rate):
        bit_alloc = np.zeros(len(sigma2))
        tmps = 0
        tmps += sigma2
        for i in range(rate):
            j = np.argmax(tmps)
            bit_alloc[j] += 1
            tmps[j] /= 4
        return bit_alloc
        
