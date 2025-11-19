"""Module for defining paths in EXASP parameter space"""
import numpy as np

class SinNPath:
    """Class for the path with functional form:
          w = wmax * k
          l = lmax * sin(k * pi)**order
    """
    def __init__(self,wmax,lmax,order):
        """Initialise the path
            wmax:  maximum value of omega
            lmax:  maximum value of lambda
            order: power order of the sin function for lambda
        """
        self.wmax = wmax
        self.lmax = lmax
        self.order = order
    
    def w(self,k):
        """Get the value of omega at k"""
        return k * self.wmax
    
    def l(self,k):
        """Get the value of lambda at k"""
        return self.lmax * np.sin(k * np.pi)**self.order
    
    def dw(self,k):
        """Get the derivative of omega at k"""
        return self.wmax
    
    def dl(self,k):
        """Get the derivative of lambda at k"""
        return self.order * np.pi * np.sin(k * np.pi)**(self.order-1) * np.cos(k * np.pi)

    def __str__(self):
        """String representation of the path"""
        return f"SinNPath(wmax={self.wmax}, lmax={self.lmax}, order={self.order})"