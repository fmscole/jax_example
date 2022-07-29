
import numpy.random as npr
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import stax
from jax.example_libraries.stax import (AvgPool, BatchNorm, Conv, Dense,
                                        FanInSum, FanOut, Flatten, GeneralConv,
                                        Identity, MaxPool, Relu, LogSoftmax,Softmax)
import numpy as np  
from dataset import getDataLoader,numpy_collate
import time

import ml_collections
def get_config():
  config = ml_collections.ConfigDict()
  config.model = 'ResNet50'
  config.dataset = 'ILSVRC2012_img_val'

  config.learning_rate = 0.1
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 100
  config.num_epochs = 40.0
  return config
config=get_config()

def ConvBlock(kernel_size, filters, strides=(2, 2)):
  ks = kernel_size
  filters1, filters2, filters3 = filters
  Main = stax.serial(
      Conv(filters1, (1, 1), strides), BatchNorm(), Relu,
      Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
      Conv(filters3, (1, 1)), BatchNorm())
  Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
  return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  def make_main(input_shape):
    return stax.serial(
        Conv(filters1, (1, 1)), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[3], (1, 1)), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


def ResNet50(num_classes):
  return stax.serial(
      GeneralConv(('NHWC', 'HWIO', 'NHWC'), 64, (7, 7),(2, 2), padding=[(3, 3), (3, 3)]),
      BatchNorm(), Relu, MaxPool((3, 3), strides=(2, 2),padding="same"),
      ConvBlock(3, [64, 64, 256], strides=(1, 1)),
      IdentityBlock(3, [64, 64]),
      IdentityBlock(3, [64, 64]),
      ConvBlock(3, [128, 128, 512]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 129]),
      IdentityBlock(3, [128, 128]),
      ConvBlock(3, [256, 256, 1024]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      ConvBlock(3, [512, 512, 2048]),
      IdentityBlock(3, [512, 512]),
      IdentityBlock(3, [512, 512]),
      AvgPool((7, 7)), 
      Flatten, 
      Dense(num_classes)
      )


rng_key = random.PRNGKey(0)  
num_classes = 1000
input_shape = (1,224, 224, 3)
init_fun, predict_fun = ResNet50(num_classes)  
outshape, init_params = init_fun(rng_key, input_shape)
print(outshape)


steps_per_epoch=40000//config.batch_size
def myschedule(steps):
    warmup_steps=config.warmup_epochs*steps_per_epoch
    cos_steps=(config.num_epochs-config.warmup_epochs)*steps_per_epoch
    alpha=0.00001
    return jnp.select([steps<warmup_steps,
                      steps<cos_steps],
                      [0.1*steps/warmup_steps,
                      0.1*(0.5*jnp.cos(jnp.pi*(steps-warmup_steps)/cos_steps)+0.5)+alpha
                      ],alpha)

    
def log_softmax(x):
  x_max=jnp.max(x,axis=-1,keepdims=True)
  x=x- jax.lax.stop_gradient(x_max) 
  x=x-jnp.log(jnp.sum(jnp.exp(x),axis=-1,keepdims=True))
  return x

def loss_fn(params, batch):
    inputs, targets = batch
    logits = predict_fun(params, inputs)
    target_class =targets
    targets=jax.nn.one_hot(targets,1000)    

    loss = jnp.mean(-jnp.sum(log_softmax(logits)*targets,axis=-1))  
    weight_penalty_params = jax.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty

    predicted_class = jnp.argmax(logits, axis=-1)
    acc_count=jnp.sum(predicted_class == target_class)    
    return loss,(loss,acc_count,target_class,predicted_class)

@jit
def accuracy(params, batch):
    inputs, targets = batch
    target_class =targets
    y=predict_fun(params, inputs)
    predicted_class = jnp.argmax(y, axis=-1)
    return jnp.sum(predicted_class == target_class)
@jit
def update(steps,params, updates, batch):
    g,ans=grad(loss_fn,has_aux=True)(params, batch)
    moments=0.9  
    lr=myschedule(steps)
    updates=jax.tree_util.tree_map(lambda x,y:lr*x+moments*y,g,updates) 
    params=jax.tree_util.tree_map(lambda x,y: x-y,params,updates)
    return params,updates,ans,lr

if __name__ == "__main__":  
  train_ds =getDataLoader(batch_size=config.batch_size,collate_fn=numpy_collate)  
  epochs=100  
  params=init_params
  updates=jax.tree_util.tree_map(lambda x:jnp.zeros_like(x),params)
  steps=0
  for e in range(epochs):
    test_ds =getDataLoader(batch_size=config.batch_size,is_train=False,collate_fn=numpy_collate)
    acc_count=0
    total=0
    for i,( x,y) in enumerate(test_ds):
      acc_count+=accuracy(params, ( x,y))
      total+=y.shape[0]
    print("")
    print("epoch:",e)
    print("test_acc=",acc_count/total)
    
    acc_count=0
    total=0
    begin=time.time()
    for i,( x,y) in enumerate(train_ds):
      steps+=1
      params,updates,(loss,batch_acc,target_class,predicted_class) ,lr= update(steps,params,updates, ( x,y))
      acc_count+=batch_acc
      total+=y.shape[0]

    print("train_acc=",acc_count/total)
    print("loss=",loss)  
    print("lr=",lr)    
    print("epoch_time=",time.time()-begin)  
    print(predicted_class)

    
      