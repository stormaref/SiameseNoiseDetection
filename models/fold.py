import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

class CustomKFoldSplitter:
    """Custom stratified k-fold cross-validation splitter with index tracking capabilities.
    
    Provides methods to maintain the mapping between validation fold indices and original dataset indices.
    """
    def __init__(self, dataset_size, labels, num_folds=10, shuffle=True):
        """Initialize the custom k-fold splitter.
        
        Args:
            dataset_size: Total number of samples in the dataset
            labels: Array of labels corresponding to the dataset
            num_folds: Number of folds for K-Fold cross-validation
            shuffle: Whether to shuffle the data before splitting into folds
        """
        self.dataset_size = dataset_size
        self.num_folds = num_folds
        self.labels = labels
        self.kf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle)
        self.folds = list(self.kf.split(np.arange(dataset_size), labels))

    def get_fold(self, fold_number):
        """Return the train and validation indices for the given fold number.

        Args:
            fold_number: The fold number (0-based)
            
        Returns:
            A tuple (train_indices, val_indices)
            
        Raises:
            ValueError: If fold_number is out of range
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        train_indices, val_indices = self.folds[fold_number]
        return train_indices, val_indices

    def get_original_index(self, fold_number, index_in_fold):
        """Return the original dataset index for a given fold and index within that fold.
        
        Args:
            fold_number: The fold number (0-based)
            index_in_fold: The index within the specified fold's validation set
            
        Returns:
            The original index in the whole dataset
            
        Raises:
            ValueError: If fold_number or index_in_fold is out of range
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        _, val_indices = self.folds[fold_number]
        
        if index_in_fold < 0 or index_in_fold >= len(val_indices):
            raise ValueError(f"Index in fold must be between 0 and {len(val_indices) - 1}.")
        
        return val_indices[index_in_fold]
    
    def get_original_indices(self, fold_number, indices_in_fold):
        """Return original dataset indices for an array of indices in a given fold.
        
        Args:
            fold_number: The fold number (0-based)
            indices_in_fold: An array of indices within the specified fold's validation set
            
        Returns:
            An array of original indices in the whole dataset
            
        Raises:
            ValueError: If fold_number or any index in indices_in_fold is out of range
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        _, val_indices = self.folds[fold_number]
        
        if any(index_in_fold < 0 or index_in_fold >= len(val_indices) for index_in_fold in indices_in_fold):
            raise ValueError(f"All indices in fold must be between 0 and {len(val_indices) - 1}.")
        
        return [val_indices[index_in_fold] for index_in_fold in indices_in_fold]
    
    def get_original_indices_as_dic(self, fold_number, indices_in_fold) -> defaultdict:
        """Return a dictionary mapping fold indices to original dataset indices.
        
        Args:
            fold_number: The fold number (0-based)
            indices_in_fold: Collection of indices within the fold's validation set
            
        Returns:
            Dictionary mapping fold indices to original dataset indices
            
        Raises:
            ValueError: If fold_number or any index in indices_in_fold is out of range
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        _, val_indices = self.folds[fold_number]
        
        if any(index_in_fold < 0 or index_in_fold >= len(val_indices) for index_in_fold in indices_in_fold):
            raise ValueError(f"All indices in fold must be between 0 and {len(val_indices) - 1}.")
        
        dic = defaultdict()
        for index_in_fold in indices_in_fold:
            real_index = val_indices[index_in_fold]
            dic[index_in_fold] = real_index
        
        return dic