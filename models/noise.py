import random

class LabelNoiseAdder:
    def __init__(self, dataset, noise_level=0.1, num_classes=10):
        self.dataset = dataset
        self.noise_level = noise_level
        self.num_classes = num_classes
        self.noisy_indices = []

    def add_noise(self):
        num_noisy_samples = int(len(self.dataset) * self.noise_level)
        self.noisy_indices = random.sample(range(len(self.dataset)), num_noisy_samples)
        
        for idx in self.noisy_indices:
            original_label = self.dataset.targets[idx]
            noisy_label = random.randint(0, self.num_classes - 1)
            
            while noisy_label == original_label:
                noisy_label = random.randint(0, self.num_classes - 1)
                
            self.dataset.targets[idx] = noisy_label

    def get_noisy_indices(self):
        return self.noisy_indices
    
    def calculate_noised_label_percentage(self, indices):
        intersection = set(indices) & set(self.noisy_indices)
        percentage = (len(intersection) / len(indices)) * 100
        print(f'{percentage}% accuracy in {len(indices)} data')
        return percentage