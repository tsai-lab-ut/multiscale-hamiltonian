import torch
from torch import nn


class VelocityVerlet(nn.Module):
    """
    Solve an initial value problem for a second-order ODE on [0, T] by velocity Verlet algorithm.
        
        d^2 x / dt^2 = A(x)
        x(0) = x0
        dx(0) / dt  = v0 
        
    """
    
    def __init__(self, A, T, nsteps=1, **kwargs):
        """
        :param A: right-hand side of the ODE
        :param T: end time 
        :param nsteps: number of time steps 
        """
        super(VelocityVerlet, self).__init__(**kwargs)

        if not callable(A):
            raise TypeError(f"Argument A should be callable, got {repr(type(A).__name__)}")
        self.A = A
        self.T = T
        self.nsteps = nsteps 
        self.h = self.T / self.nsteps
        
    def solve(self, v0, x0, retfull=False):
        """
        Integrate the ODE from 0 to T given the initial state (v0, x0). 

        :param retfull: whether or not to return solutions at all time points
        """
        
        # Initialize solution 
        x = x0
        v = v0
        res = [(v0, x0)]
        
        # Integrate using velocity Verlet 
        for _ in range(self.nsteps): 
            v_mid = v + 0.5 * self.h * self.A(x)
            x = x + v_mid * self.h
            v = v_mid + 0.5 * self.h * self.A(x)
            res.append((v, x))
        
        # Return solution 
        if retfull:
            return res
        else:
            return v, x
    
    def forward(self, u0):
        """
        Integrate the ODE from 0 to T given the initial state u0 = (v0, x0). 
        """
        v0, x0 = u0.chunk(2, dim=-1)
        v, x = self.solve(v0, x0)
        return torch.cat((v, x), dim=-1)

    def extra_repr(self):
        """
        Set the extra representation of the module.
        """
        return f"T={self.T}, nsteps={self.nsteps}, h={self.h}"
