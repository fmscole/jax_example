import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf 
import time
from torch.utils.data import DataLoader
from dataset import train_dataset,val_dataset,test_loader
import pandas as pd
import os
from model import CNN
import pickle
import pywt
data_root="/media/fms/E0702E72702E4F98/dataset/naowen/"


batch_size=100
num_epochs=30
class_nums=95




def create_train_state(rng):
  cnn = CNN()
  variables=cnn.init(rng, jnp.ones([2,100, 65, 500, 1]))
  params = variables['params']
  batch_stats=variables['batch_stats']

  
  def create_learning_rate_fn(base_learning_rate: float,steps_per_epoch: int):
    warmup_epochs=5    
    warmup_fn = optax.linear_schedule(init_value=0.000001, end_value=base_learning_rate,transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,decay_steps=cosine_epochs * steps_per_epoch,alpha=0.0001)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn

  steps_per_epoch=60000//batch_size
  schedule=create_learning_rate_fn(base_learning_rate=0.01,steps_per_epoch=steps_per_epoch)
  tx = optax.sgd(learning_rate=schedule,momentum= 0.90)
  state=train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
  return state,batch_stats

@jax.jit
def predict(state, batch_stats,image_i):
  logits= state.apply_fn({'params': state.params,"batch_stats":batch_stats},image_i,is_training=False)
  return logits
def val(state, batch_stats,test_ds):
  
  accuracy=0
  total=0
  for x,y in test_ds:
    batch_images = np.array(x)
    x1,x2=pywt.dwt(batch_images, 'haar')
    # print(x1.shape,x2.shape)
    x1=x1[...,None]
    x2=x2[...,None]
    batch_images=(x1,x2)
    batch_labels = np.array(y)
    image_i=batch_images
    label_i=batch_labels
    logits= predict(state, batch_stats,image_i)
    accuracy += jnp.sum(jnp.argmax(logits, -1) == label_i) 
    total+=y.shape[0] 
  return accuracy/total

def save_weights(weights, filename):
    bytes_output = flax.serialization.to_bytes(weights)
    pickle.dump(bytes_output, open(filename, 'wb'))

def load_weights(weights, filename):
    pkl_file = pickle.load(open(filename, 'rb'))
    trained_wts = flax.serialization.from_bytes(weights, pkl_file)
    return trained_wts

def test(state, batch_stats,test_ds):  
    preds=[]
    images=[]
    for x,y in test_ds:
        batch_images = np.array(x)
        x1,x2=pywt.dwt(batch_images, 'haar')
        # print(x1.shape,x2.shape)
        x1=x1[...,None]
        x2=x2[...,None]
        batch_images=(x1,x2)  
        image_i=batch_images
        logits= predict(state, batch_stats,image_i)
        pre=jnp.argmax(logits, -1)+1
        preds=preds+list(pre)
        images=images+list(y)
    return preds,images
def creat_result(state, batch_stats,test_loader):
    pre_list,mg_list=test(state, batch_stats,test_loader)
    test_image = pd.read_csv(data_root+'/Testing_Info.csv')
    test_info = test_image['SubjectID'].values
    
    img_test = pd.DataFrame(mg_list)
    img_test = img_test.rename(columns = {0:"EpochID"})

    img_pre = pd.DataFrame(mg_list)    
    img_pre['SubjectID'] = pre_list
    pre_info = img_pre['SubjectID'].values

    print("begin")
    result_cnn = list()
    for i,j,k1,k2 in zip(test_info, pre_list,mg_list,img_test.values):
      if k1!=k2:
        print(i,j,k1,k2)
      else:
        if i == 'None':
            result_cnn.append(j)
        elif int(i[4:])==j :
            result_cnn.append(int(1))
        else:
            result_cnn.append(int(0))

    img_test['Prediction'] = result_cnn
    img_test.to_csv('result_dwt_resnet.csv', index=False)
    print("finish")

def train_and_evaluate() :  
    rng = jax.random.PRNGKey(0)  
    rng, init_rng = jax.random.split(rng)
    state,batch_stats = create_train_state(init_rng)
    best=0
    filename="cnn_best2.npy"
    if os.path.exists(filename):
      weight={"state":state,"batch_stats":batch_stats,"p":best}
      weight=load_weights(weight, filename)
      state=weight["state"]
      batch_stats=weight["batch_stats"]
      best=weight["p"]
      print(f"load best={best}")

   
    test_ds= DataLoader(val_dataset, batch_size=100,num_workers=4,shuffle=True,collate_fn=None)
    begin=time.time()
    p=val(state, batch_stats,test_ds)
    print(f"time:{time.time()-begin:0.2f}")
    print(f"test_acc={p:0.2f}",end=" ")
    print("")

    start =time.time()
    creat_result(state, batch_stats,test_loader)
    print("time:",time.time()-start)
train_and_evaluate() 