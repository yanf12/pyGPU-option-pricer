from numba import cuda
import numpy as np
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
# Example PyPI (Python Package Index) Package

class GPUpricer(object):

    def __init__(self,T,K, S0, sigma, mu, r,  N_STEPS, N_PATHS):
        """
        :param T: Time to expiration
        :param K: Strike price
        :param S0: Initial spot price
        :param sigma: Volatility
        :param mu: Drift rate, equals to r-q under risk-free measure
        :param r: risk-free rate
        :param N_STEPS: Number of time steps in each path
        :param N_PATHS: Number of paths used for Monte-Carlo simulation
        """
        self.T = T
        self.K = K
        self.S0 = S0
        self.sigma = sigma
        self.mu = mu
        self.r = r
        self.N_STEPS = N_STEPS
        self.N_PATHS = N_PATHS
        self.d_s = np.zeros(N_PATHS, dtype=np.float32) # Stores the result of each path

    def _get_normal_paths(self):
        self.normals = np.random.normal(0, 1, self.N_STEPS * self.N_PATHS).astype(np.float32)

    @staticmethod
    @cuda.jit
    def get_pay_off(rng_states, d_s, T, mu, sigma, N_STEPS, N_PATHS, S0, K, r):
        idx = cuda.grid(1)
        #print(idx)
        dt = T / N_STEPS
        discount = math.exp(-r * T)

        if idx < N_PATHS:
            s_curr = 0.0
            for n in range(N_STEPS):
                s_curr += mu * dt - 0.5 * (sigma ** 2) * dt + sigma * math.sqrt(dt) * xoroshiro128p_normal_float32(rng_states, idx)
            ST = S0 * math.exp(s_curr)
            payoff = ST - K if ST > K else 0.0
            d_s[idx] = discount * payoff

    def simulate(self):
        d_s = cuda.device_array(self.N_PATHS, dtype=np.float32)
        rng_states = create_xoroshiro128p_states(self.N_PATHS, seed=1)
        block_size = 256
        grid_size = (self.N_PATHS + block_size - 1) // block_size
        self.get_pay_off[grid_size, block_size](rng_states, d_s, self.T, self.mu, self.sigma, self.N_STEPS, self.N_PATHS, self.S0, self.K, self.r)
        self.d_s = d_s.copy_to_host()
        return self.d_s.mean()





if __name__=="__main__":
    T, K, S0, sigma, mu, r = 1.0, 110.0, 120.0, 0.35, 0.05, 0.05
    N_STEPS, N_PATHS = 360, 10000000
    vanilla_example = GPUpricer(T, K, S0, sigma, mu, r, N_STEPS, N_PATHS)
    print(vanilla_example.simulate())


