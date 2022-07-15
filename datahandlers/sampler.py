import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, task_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(task_vector.size(0) / batch_size)
        self.task_vector = task_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        X = torch.randn(self.task_vector.size(0),2).numpy()
        y = self.task_vector.numpy()
        s.get_n_splits(X, y)

        indices = []
        for _, test_index in s.split(X, y):
            indices = np.hstack([indices, test_index])

        return indices.astype('int')

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.task_vector)