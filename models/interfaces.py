from abc import ABC, abstractmethod

class NoiseAdder(ABC):
    """Abstract base class for adding noise to dataset labels."""
    
    @abstractmethod
    def add_noise(self, norm_std=0.1, seed=21):
        """Add noise to labels with given standard deviation and random seed."""
        pass
        
    @abstractmethod
    def get_noisy_indices(self):
        """Return indices of samples that have noisy labels."""
        pass
    
    @abstractmethod
    def calculate_noised_label_percentage(self, indices):
        """Calculate the percentage of noisy labels within given indices."""
        pass
    
    @abstractmethod
    def report(self, indices):
        """Generate a report about noise for the specified indices."""
        pass
    
    @abstractmethod    
    def ravel(self, indices):
        """Flatten/ravel the indices for analysis."""
        pass