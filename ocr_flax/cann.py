import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import pickle

class CANN(nn.Module):
  class_nums:int
  @nn.compact
  def __call__(self, x,is_training:bool=True):   
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)# [B,64,W,32]
    
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2),padding="same")# [B,64,W/2,16]
    
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)# [B,128,W/2,16]
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2),padding="same")# [B,128,W/4,8]
    
    x = nn.Conv(features=256, kernel_size=(3, 3))(x)# [B,256,W/4,8]
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.Conv(features=256, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(1, 2),padding="same")# [B,256,W/4,4]
    
    x = nn.Conv(features=512, kernel_size=(3, 3))(x)# [B,512,W/4,4]
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.Conv(features=512, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(1, 2),padding="same") # [B,512,W/4,2]
    
    x = nn.Conv(features=512, kernel_size=(2, 2),padding="valid")(x)# [B,512,W/4,1]?
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x=x.reshape(x.shape[0],x.shape[1],-1)
      
    q = nn.Dense(features=512)(x)    
    k = nn.Dense(features=512)(x)
    v = nn.Dense(features=512)(x)
    
    qk=jnp.einsum("btk,bsk->bts",q,k)/32
    qk=nn.softmax(qk,axis=-1)    
    x=jnp.einsum("btk,bts->bsk",v,qk)
    
    x=nn.Dense(features=self.class_nums)(x)  
  
    return x

if __name__ =="__main__":
    cnn = CRNN(class_nums=9)
    variables=cnn.init(jax.random.PRNGKey(0), jnp.ones([2,512, 32, 1]))