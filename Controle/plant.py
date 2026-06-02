import numpy as np

class Plant:
    
    def set_order_one(self, dt, tau, k):
        self.a_one = np.exp(-dt/tau)
        self.b_one = k * (1 - self.a_one)

        self.y_one = 0.0 
    
    def order_one(self, u):
        self.y_one = self.a_one * self.y_one + self.b_one * u

        return self.y_one
