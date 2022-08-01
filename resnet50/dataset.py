
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
'''
ILSVRC2012_img_val/
|---val_list.txt
|---images/
'''
class ImageNetBaseSet(Dataset):
    def __init__(self,data_root=r"/mnt/e/dataset/ILSVRC2012_img_val"):               
        data_txt=data_root+r'/val_list.txt'
        self.transforms=transforms   

        files=open(data_txt,'r')
        content=files.readlines()
        files.close()

        self.image_paths=[]
        self.labels=[]
        for i in content:
            f,cls=i.split(" ")
            cls=int(cls)
            self.labels.append(cls)
            self.image_paths.append(data_root+"/images/"+f)                                   

    def __getitem__(self, index):
        image_file=self.image_paths[index]
        im=Image.open(image_file).convert("RGB")
        cls=self.labels[index] 
        return im,cls

    def __len__(self):
        return len(self.labels)

class ImageNetSet(Dataset):
    def __init__(self,baseset,transforms=None):               
        self.baseset=baseset
        self.transforms=transforms
    def __getitem__(self, index):
        im, cls=self.baseset.__getitem__(index)               
        if self.transforms:
            try:
                x=self.transforms(im)
            except:
                print(x.shape)
        else:
            im=im.resize((224,224))
            x=np.array(im)/255.0
        return x,cls
    def __len__(self):
        return self.baseset.__len__()

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    if isinstance(batch, (tuple,list)):
      if not isinstance(batch[0], int):
        batch=[np.array(im).transpose(1,2,0) for im in batch]
    return np.array(batch)


basedataset=ImageNetBaseSet()
dataset40000, dataset10000 =random_split(basedataset, [40000, 10000])


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
trainset=ImageNetSet(dataset40000,transforms=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

valset=ImageNetSet(dataset10000,transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))



def getDataLoader(batch_size=100,is_train=True,num_workers=16,shuffle=True,collate_fn=None):
    if is_train:
        training_generator = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,shuffle=shuffle,collate_fn=collate_fn)
        return training_generator
    else: 
        test_dataset1, test_dataset =random_split(valset, [5000, 5000])
        val_generator = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,collate_fn=collate_fn)
        return val_generator

