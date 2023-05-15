from numba import cuda
import numpy as np
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
        self.N_steps = N_STEPS
        self.N_PATHS = N_PATHS
        self.d_s = np.zeros(N_PATHS, dtype=np.float32) # Stores the result of each path

    def _get_normal_paths(self):
        self.normals = np.random.normal(0, 1, self.N_STEPS * self.N_PATHS).astype(np.float32)

    def get_pay_off(self,i):
        s_curr = 0
        discount = np.exp(-self.r * self.T)
        dt = self.T / self.N_STEPS
        for n in range(self.N_STEPS):
            s_curr += self.mu * dt - 0.5 * (self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * self.normals[i + n * self.N_PATHS]
        ST = self.S0 * np.exp(s_curr)
        payoff = ST - self.K if ST > self.K else 0.0
        self.d_s[i] = discount * payoff

    @cuda.jit
    def simulate(self):
        self._get_normal_paths()
        block_size = 256
        grid_size = (self.N_PATHS + block_size - 1) // block_size
        self.get_pay_off[grid_size, block_size]()
        return self.d_s.mean()


if __name__=="__main__":
    T, K, S0, sigma, mu, r = 1.0, 110.0, 120.0, 0.35, 0.05, 0.05
    N_STEPS, N_PATHS = 360, 100000
    vanilla_example = GPUpricer(T, K, S0, sigma, mu, r, N_STEPS, N_PATHS)


