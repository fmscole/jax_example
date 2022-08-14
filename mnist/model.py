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
import pickle


batch_size=100
num_epochs=30
class_nums=95
class CNN(nn.Module):
  @nn.compact
  def __call__(self, x,is_training:bool=True):   
    x1,x2=x[0],x[1]
    x1 = nn.Conv(features=32, kernel_size=(3, 3))(x1)
    x1 = nn.BatchNorm(use_running_average=not is_training)(x1)
    x1 = nn.relu(x1)
    x1 = nn.avg_pool(x1, window_shape=(11, 11), strides=(1, 1),padding="same")

    x2 = nn.Conv(features=32, kernel_size=(3, 3))(x2)
    x2 = nn.BatchNorm(use_running_average=not is_training)(x2)
    x2 = nn.relu(x2)
    x2 = nn.avg_pool(x2, window_shape=(11, 11), strides=(1, 1),padding="same")
    

    q = nn.Dense(features=1)(x1)    
    k = nn.Dense(features=1)(x1)
    v = nn.Dense(features=1)(x1)
    bef=jnp.einsum("bmnk,bmjk->bnj",q,k)/8
    x3=nn.softmax(bef,axis=-1)
    x3=x3[...,None]
    aft=jnp.einsum("bmnk,bnjk->bmj",v,x3)
    x1=aft
    
    q = nn.Dense(features=1)(x2)    
    k = nn.Dense(features=1)(x2)
    v = nn.Dense(features=1)(x2)
    bef=jnp.einsum("bmnk,bmjk->bnj",q,k)/8
    x3=nn.softmax(bef,axis=-1)
    x3=x3[...,None]
    aft=jnp.einsum("bmnk,bnjk->bmj",v,x3)
    x2=aft
    # k=k.transpose(0,2,1,3)
    # t=q*k/16
    # s=nn.softmax(t,axis=2)    
    # x=s*v
    # x=jnp.max(x,axis=(2,3))
    # print(x.shape)
    x=jnp.stack((x1,x2),axis=-1)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=class_nums)(x)
    return x

cnn = CNN()
variables=cnn.init(jax.random.PRNGKey(0), jnp.ones([2,100, 65, 500,1]))


class SimpleScan(nn.Module):
    @nn.compact
    def __call__(self, x,is_training:bool=True):
        LSTM = nn.scan(nn.LSTMCell,
                    variable_broadcast="params",
                    split_rngs={"params": False},
                    in_axes=2,
                    out_axes=2)    
        
        x = nn.Conv(features=32, kernel_size=(3,))(x)
        y = nn.BatchNorm(use_running_average=not is_training)(x)
        x = nn.relu(x)      
        x = nn.avg_pool(x, window_shape=(2, ), strides=(2, ))

        ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],), 100)
        ch, x=LSTM()(ch, x)
                
        # xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)

        ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],), 100)
        ch, x=LSTM()(ch, x)        
        
        # # xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)
        # ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],), 200)
        # ch, x=LSTM()(ch, x)
        
        # # xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)
        # ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (x.shape[0],), 200)
        # ch, x=LSTM()(ch, x)
        # # xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)
        # # xs = nn.BatchNorm(use_running_average=not is_training)(xs)

        x=x.reshape(x.shape[0],-1)
        x=nn.Dense(features=class_nums)(x)        
        return x