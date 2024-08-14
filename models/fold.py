import numpy as np
from sklearn.model_selection import KFold

class CustomKFoldSplitter:
    def __init__(self, dataset_size, num_folds=10, shuffle=True, random_state=None):
        self.dataset_size = dataset_size
        self.num_folds = num_folds
        self.kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
        self.folds = list(self.kf.split(np.arange(dataset_size)))

    def get_fold(self, fold_number):
        """
        Returns the train and validation indices for the given fold number.

        :param fold_number: The fold number (0-based).
        :return: A tuple (train_indices, val_indices).
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        train_indices, val_indices = self.folds[fold_number]
        return train_indices, val_indices

    def get_original_index(self, fold_number, index_in_fold):
        """
        Returns the original index in the whole dataset for a given fold and index within that fold.
        
        :param fold_number: The fold number (0-based).
        :param index_in_fold: The index within the specified fold's validation set.
        :return: The original index in the whole dataset.
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        _, val_indices = self.folds[fold_number]
        
        if index_in_fold < 0 or index_in_fold >= len(val_indices):
            raise ValueError(f"Index in fold must be between 0 and {len(val_indices) - 1}.")
        
        return val_indices[index_in_fold]
    
    def get_original_indices(self, fold_number, indices_in_fold):
        """
        Returns an array of original indices in the whole dataset for an array of indices in a given fold.
        
        :param fold_number: The fold number (0-based).
        :param indices_in_fold: An array of indices within the specified fold's validation set.
        :return: An array of original indices in the whole dataset.
        """
        if fold_number < 0 or fold_number >= self.num_folds:
            raise ValueError(f"Fold number must be between 0 and {self.num_folds - 1}.")
        
        _, val_indices = self.folds[fold_number]
        
        if any(index_in_fold < 0 or index_in_fold >= len(val_indices) for index_in_fold in indices_in_fold):
            raise ValueError(f"All indices in fold must be between 0 and {len(val_indices) - 1}.")
        
        return [val_indices[index_in_fold] for index_in_fold in indices_in_fold]
