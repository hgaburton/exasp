"""Module for handling measurement outcomes with uncertainties."""
import numpy as np

def add_measurements(lmeas,const=0):
    """ Add a list of measurement outcomes with uncertainties
        Args:
            lmeas: List of measurement outcomes
            const: Constant to add to the sum
        Returns:
            x: Total value
            dx: Total uncertainty
    """
    outcomes = np.zeros((len(lmeas),2))
    # Collect individual uncertainty
    for i, meas in enumerate(lmeas):
        outcomes[i,:] = meas.value()
    # Get total value 
    x = np.sum(outcomes[:,0]) + const
    # Get total uncertainty
    dx = np.sqrt(np.sum(np.power(outcomes[:,1],2)))
    return float(x), float(dx)

def multiply_measurements(lmeas,prefac=1):
    """ Multiply a list of measurement outcomes with uncertainties
        Args:
            lmeas: List of measurement outcomes
            prefac: Prefactor to multiply to the product
        Returns:
            x: Total value
            dx: Total uncertainty
    """
    nmeas = len(lmeas)
    # Collect individual uncertainty
    outcomes = np.zeros((nmeas,2))
    for i, meas in enumerate(lmeas):
        outcomes[i,:] = meas.value()
    # get total value
    x = prefac*np.prod(outcomes[:,0])
    # Get total variances
    dx = x*np.sqrt(np.sum(np.power(outcomes[:,1]/outcomes[:,0], 2))) 
    return float(x), float(dx)

class measurement_outcome:
    """ Class for handling measurement outcomes with uncertainties."""
    def __init__(self):
        self.outcomes = dict()

    def add_outcome(self, value, freq):
        """ Add an outcome to the measurement outcome dictionary
            Args:
                value: The value of the outcome
                freq : Frequence of an outcome
        """
        if value not in self.outcomes:
            self.outcomes[value] = int(0)
        self.outcomes[value] += int(freq)
    
    def __repr__(self):
        return str(self.outcomes)

    @property
    def total_outcomes(self):
        """ Total number of outcomes recorded
            Returns:
                total_outcomes: Total number of outcomes
        """
        return sum(self.outcomes.values())
    
    def value(self):
        """ Compute mean and standard error of the outcomes
            Returns:
                mean: The mean of the outcomes
                stderr: The standard error of the outcomes
        """
        if self.total_outcomes == 0:
            return np.nan, np.nan
        # Compute the mean
        mean = sum([k * v for k, v in self.outcomes.items()]) / self.total_outcomes
        # Compute the variance
        variance = sum([(k - mean) ** 2 * v for k, v in self.outcomes.items()]) / self.total_outcomes
        # Compute the standard error
        return float(mean), float(np.sqrt(variance / self.total_outcomes))
