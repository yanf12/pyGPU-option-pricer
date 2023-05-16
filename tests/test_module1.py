import unittest

from gpupricer.module1 import GPUpricer


class TestSimple(unittest.TestCase):

    def test_1(self):
        T, K, S0, sigma, mu, r = 1.0, 110.0, 120.0, 0.35, 0.05, 0.05
        N_STEPS, N_PATHS = 360, 5000000
        vanilla_example = GPUpricer(T, K, S0, sigma, mu, r, N_STEPS, N_PATHS)
        res = vanilla_example.simulate()
        self.assertEqual(res,24.45)



if __name__ == '__main__':
    unittest.main()

