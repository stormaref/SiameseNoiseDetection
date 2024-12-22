from abc import ABC, abstractmethod

class NoiseAdder(ABC):
    @abstractmethod
    def add_noise(self, norm_std=0.1, seed=21):
        pass
        
    @abstractmethod
    def get_noisy_indices(self):
        pass
    
    @abstractmethod
    def calculate_noised_label_percentage(self, indices):
        pass
    
    @abstractmethod
    def report(self, indices):
        pass
    
    @abstractmethod    
    def ravel(self, indices):
        pass