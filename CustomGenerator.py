import numpy as np
import tensorflow as tf



        

class ReturnsDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, data,
                 shuffle_stocks=True,
                 shuffle_batches=True,
                 extra_dim=False,
                 seed=42):
        
        self.data = data.copy()
        self.shuffle_stocks = shuffle_stocks
        self.shuffle_batches = shuffle_batches
        self.extra_dim = extra_dim
        self.seed = seed
        self.batches = self.data.shape[0]
        self.length = self.data.shape[1]
        self.stocks = self.data.shape[2]
        print('Number of training batches: ', self.batches)
        print('Length of each batch: ', self.length)
        print('Number of stocks: ', self.stocks)
        
        if self.shuffle_batches:
            np.random.seed(self.seed)
            indexes = np.random.randint(0, self.batches, self.batches)
            self.indexes = iter(indexes)
        else:
            self.indexes = iter(range(self.batches))
        self.index = next(self.indexes)
           
            
    def on_epoch_end(self):
        self.index = next(self.indexes)
        # print('New batch index: ', self.index)
            
    
    def __getitem__(self, *args, **kwargs):
        batch = self.data[self.index, :, :]
        batch = np.transpose(batch)
        if self.shuffle_stocks:
            batch = batch[:, np.random.permutation(batch.shape[1])]
        if self.extra_dim:
            batch = np.expand_dims(batch, axis=2)
        print('Batch shape: ', batch.shape)
        return batch
    
    def __len__(self):
        return self.batches 
    
class CodeBookDataGen(tf.keras.utils.Sequence):
    def __init__(self,codebook,
                shuffle_batches=True,
                seed=42):
        self.codebook = codebook.copy()
        self.shuffle_batches = shuffle_batches
        self.seed = seed
        self.batches = self.codebook.shape[0]
        self.latent = self.codebook.shape[2]
        self.stocks = self.codebook.shape[1]
        print('Number of training batches: ', self.batches)
        print('Number of stocks: ', self.stocks)
        print('Number of latent variables: ', self.latent)
        if self.shuffle_batches:
            np.random.seed(self.seed)
            indexes = np.random.randint(0, self.batches, self.batches)
            self.indexes = iter(indexes)
        else:
            self.indexes = iter(range(self.batches))
        self.index = next(self.indexes)
    
    def on_epoch_end(self):
        self.index = next(self.indexes)
    
    def __getitem__(self, *args, **kwargs):
        batch = self.codebook[self.index, :, :]
        
        return batch, batch
    def __len__(self):
        return self.batches