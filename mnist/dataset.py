
import torch
torch.device("cpu")

from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import scipy.io as scio
from sklearn.utils import shuffle
import os
import pandas as pd
# import paddle


data_root="/mnt/c/Users/fms/Downloads/naowen/"
task=[1,2,3,4,5,6,7,8,9,10,11,12,13]
# task=[1]
# 读取数据
train_images = pd.read_csv(data_root+'Enrollment_Info.csv')
val_images = pd.read_csv(data_root+'Calibration_Info.csv')
test_images = pd.read_csv(data_root+'Testing_Info.csv')

train_images =train_images[train_images["Task"].isin(task)]
val_images =val_images[val_images["Task"].isin(task)]
test_images =test_images[test_images["Task"].isin(task)]

train_images = shuffle(train_images, random_state=0)
val_images = shuffle(val_images)
# 划分训练集和校验集

train_image_list = train_images
val_image_list = val_images

df = train_image_list
train_image_path_list = train_image_list['EpochID'].values
train_label_list = train_image_list['SubjectID'].values


val_image_path_list = val_image_list['EpochID'].values
val_label_list = val_image_list['SubjectID'].values

test_image_path_list = test_images['EpochID'].values

class MyDataset(Dataset):
    def __init__(self, train_img_list=train_image_path_list, 
                        val_img_list=val_image_path_list, 
                        test_img_list=test_image_path_list, 
                        train_label_list=train_label_list, 
                        val_label_list=val_label_list, 
                        mode='train'):
        super(MyDataset, self).__init__()
        self.img = []
        self.label = []
        self.mode=mode
        # 借助pandas读csv的库
        self.train_images = train_img_list
        self.val_images = val_img_list
        self.test_images = test_img_list

        self.train_label = train_label_list
        self.val_label = val_label_list
        if self.mode == 'train':
            # 读train_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append(data_root+'Enrollment/'+img+'.mat')
                self.label.append(np.array(int(la[4:]) - 1, dtype=np.int64))
        elif self.mode == 'test':
            for img in self.test_images:
                self.img.append(data_root+'Testing/'+img+'.mat')
                self.label.append(img)                
        else:            
            for img,la in zip(self.val_images, self.val_label):
                self.img.append(data_root+'Calibration/'+img+'.mat')
                self.label.append(np.array(int(la[4:]) - 1, dtype=np.int64))

    def load_eeg(self, eeg_path):
        data = scio.loadmat(eeg_path)
        return data['epoch_data']

    def __getitem__(self, index):
        eeg_data = self.load_eeg(self.img[index])
        data_max=np.max(eeg_data,keepdims=True)
        data_min=np.min(eeg_data,keepdims=True)
        eeg_data=(eeg_data-data_min)/(data_max-data_min)
        eeg_data=eeg_data

        # if self.mode == 'test':
        #     eeg_label=0
        # else:
        eeg_label = self.label[index]
        # label = paddle.to_tensor(label)
        
        return eeg_data,eeg_label

    def __len__(self):
        return len(self.img)



class ImageNetSet(Dataset):
    def __init__(self,baseset,transforms=None):               
        self.baseset=baseset
        self.transforms=transforms
    def __getitem__(self, index):
        im, cls=self.baseset.__getitem__(index)               
        if self.transforms:
            try:
                x=self.transforms(im)
            except :
                print(x.shape)
        else:
            im=im.resize((224,224))
            x=np.array(im)/255.0
        return x,cls
    def __len__(self):
        return self.baseset.__len__()

train_dataset = MyDataset( mode='train')
val_dataset = MyDataset(mode='val')

train_loader = DataLoader(train_dataset, batch_size=50,num_workers=8,shuffle=True,collate_fn=None)
val_loader = DataLoader(val_dataset, batch_size=50,num_workers=8,shuffle=True,collate_fn=None)


test_dataset = MyDataset(mode='test')
test_loader = DataLoader(test_dataset, batch_size=50,num_workers=8,shuffle=False,collate_fn=None)



# basedataset=MyDataset( mode='train')
# train_dataset, val_dataset =random_split(basedataset, [basedataset.__len__()-5000, 5000])


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# # train_dataset=ImageNetSet(dataset40000,transforms=transforms.Compose([
# #             transforms.ToTensor(),
# #             normalize,
# #         ]))

# # val_dataset=ImageNetSet(dataset10000,transforms=transforms.Compose([
# #             transforms.ToTensor(),
# #             normalize,
# #         ]))



# train_loader = DataLoader(val_dataset, batch_size=10,num_workers=4,shuffle=True,collate_fn=None)
# val_loader = DataLoader(val_dataset, batch_size=10,num_workers=4,shuffle=True,collate_fn=None)

# # def getDataLoader(batch_size=100,is_train=True,num_workers=8,shuffle=True,collate_fn=None):
# #     if is_train:
# #         training_generator = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=shuffle,collate_fn=collate_fn)
# #         return training_generator
# #     else: 
# #         test_dataset1, test_dataset =random_split(valset, [5000, 5000])
# #         val_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,collate_fn=collate_fn)
# #         return val_generator
import lightgbm
class LGBSequence(lightgbm.Sequence):
    def __init__(self, train_img_list=train_image_path_list, 
                        val_img_list=val_image_path_list, 
                        test_img_list=test_image_path_list, 
                        train_label_list=train_label_list, 
                        val_label_list=val_label_list, 
                        batch_size=1,
                        mode='train',xy="x"):
        self.batch_size = batch_size
        self.train_len=5000
        self.xy=xy
        self.img = []
        self.label = []
        self.mode=mode
        # 借助pandas读csv的库
        self.train_images = train_img_list
        self.val_images = val_img_list
        self.test_images = test_img_list

        self.train_label = train_label_list
        self.val_label = val_label_list
        if self.mode == 'train':
            # 读train_images的数据
            for img,la in zip(self.train_images, self.train_label):
                self.img.append(data_root+'Enrollment/'+img+'.mat')
                self.label.append(int(la[4:]) - 1)
        elif self.mode == 'test':
            for img in self.test_images:
                self.img.append(data_root+'Testing/'+img+'.mat')
                self.label.append(img)                
        else:            
            for img,la in zip(self.val_images, self.val_label):
                self.img.append(data_root+'Calibration/'+img+'.mat')
                self.label.append(int(la[4:]) - 1)

    def load_eeg(self, eeg_path):
        data = scio.loadmat(eeg_path)
        return data['epoch_data']
    def getY(self):
        if self.mode == 'train':
            return self.label[0:self.train_len]
        return self.label

    def __getitem__(self, index):
        if self.xy=="x":
            img_list=self.img[index]
            if isinstance(img_list, (tuple,list)): 
                eeg_datas=[]
                for im in img_list:
                    eeg_data = self.load_eeg(im)
                    data_max=np.max(eeg_data,keepdims=True)
                    data_min=np.min(eeg_data,keepdims=True)
                    eeg_data=(eeg_data-data_min)/(data_max-data_min)
                    eeg_data=eeg_data.reshape(-1)
                    eeg_datas.append(eeg_data)
                eeg_data=eeg_datas
            else:
                eeg_data = self.load_eeg(img_list)
                data_max=np.max(eeg_data,keepdims=True)
                data_min=np.min(eeg_data,keepdims=True)
                eeg_data=(eeg_data-data_min)/(data_max-data_min)
                eeg_data=eeg_data.reshape(-1)

            
            return eeg_data
        else:
            eeg_label = self.label[index]
            return eeg_label

    def __len__(self):
        if self.mode == 'train':
            return self.train_len
        return len(self.img)

x_train_lgb = LGBSequence( mode='train',xy="x")
# y_train_lgb = LGBSequence( mode='train',xy="y")
x_val_lgb = LGBSequence(mode='val',xy="x")
# y_val_lgb = LGBSequence(mode='val',xy="y")

# lgb_train = lightgbm.Dataset(x_train_lgb, y_train_lgb)
if __name__=="__main__":
    print(x_train_lgb.__len__())
    print(x_train_lgb.getY())