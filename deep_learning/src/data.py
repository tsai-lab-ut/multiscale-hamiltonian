import logging
import os 
import pandas as pd
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl


logger = logging.getLogger(__name__)

def split_dataset(ds, train_fraction, seed):
    n_full = len(ds)
    n_train = int(train_fraction*n_full)
    n_test = n_full - n_train 
    ds_train, ds_test = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    return ds_train, ds_test 


def get_dataset(data_dir, sequence_len, dtype):

    filenames = [f"U{n}.csv" for n in range(sequence_len+1)]
    data = []
    for fname in filenames: 
        u = pd.read_csv(os.path.join(data_dir, fname)).to_numpy()
        u = torch.tensor(u, dtype=dtype)
        data.append(u)
    ds = TensorDataset(*data)
    return ds


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, sequence_len, dtype="float64", 
                 batch_size=100, num_workers=4, pin_memory=True):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir 
        self.sequence_len = sequence_len 
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage):
        self.ds_train = get_dataset(self.train_dir, self.sequence_len, self.dtype)
        self.ds_test = get_dataset(self.test_dir, self.sequence_len, self.dtype)
    
        logger.info("U_n (n=0,1,...,{0}) train: {1}".format(len(self.ds_train[:])-1, self.ds_train[:][0].shape))
        logger.info("U_n (n=0,1,...,{0}) test: {1}".format(len(self.ds_test[:])-1, self.ds_test[:][0].shape))

    def train_dataloader(self):
        # if self.trainer.current_epoch < 800:
        #     batch_size = 200
        # elif self.trainer.current_epoch < 2000:
        #     batch_size = 400
        # elif self.trainer.current_epoch < 4400:
        #     batch_size = 800
        # else:
        #     batch_size = self.batch_size
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
