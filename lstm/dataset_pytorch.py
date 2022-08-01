
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
'''
7278
(6500, 6) (700, 6)
'''

class StocksSet(Dataset):
    def __init__(self,is_train:bool=True):               
        data = pd.read_csv('IBM.csv')
        all_data = data.iloc[:, 1:7]
        all_data=all_data.to_numpy()
        self.data_max=np.max(all_data,axis=0)
        self.data_min=np.min(all_data,axis=0)
        self.train_len=6500
        self.test_len=700
        self.all_data_scaled=(all_data-self.data_min)/(self.data_max-self.data_min)                                   
        self.is_train=is_train
        
    def __getitem__(self, index):
        if self.is_train:
            x=self.all_data_scaled[index:index+60]
            y=self.all_data_scaled[index+60]
        else:
            index+=6500
            x=self.all_data_scaled[index:index+60]
            y=self.all_data_scaled[index+60]
        return x,y

    def __len__(self):
        if self.is_train:
            return self.train_len
        else:
            return self.test_len

trainset=StocksSet()
test_dataset=StocksSet(is_train=False)
training_generator=DataLoader(trainset, batch_size=100, num_workers=4)
test_generator = DataLoader(test_dataset, batch_size=100, num_workers=4)

