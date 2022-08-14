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
import os
from model import CNN,SimpleScan
import pywt
batch_size=100
num_epochs=30
class_nums=95


@jax.jit
def apply_model(state, images, labels,old_batch_stats):
  def loss_fn(params,old_batch_stats):
    logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_batch_stats}, images,is_training=True, mutable=['batch_stats'])
    one_hot = jax.nn.one_hot(labels, class_nums)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

    weight_penalty_params = jax.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty

    return loss, (logits,mutated_vars['batch_stats'])    
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits,new_batch_stats)), grads = grad_fn(state.params,old_batch_stats)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy,new_batch_stats

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def get_datasets():
    
#   with tf.device('/cpu:0'):
#     # ds_builder = tfds.builder('fashion_mnist')
#     ds_builder = tfds.builder('mnist')
#     ds_builder.download_and_prepare()
#     train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
#     test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
#     train_ds['image'] = jnp.float32(train_ds['image']) / 255.
#     test_ds['image'] = jnp.float32(test_ds['image']) / 255.
#     return train_ds, test_ds

    train_loader = DataLoader(train_dataset, batch_size=50,num_workers=4,shuffle=True,collate_fn=None)
    val_loader = DataLoader(val_dataset, batch_size=50,num_workers=4,shuffle=True,collate_fn=None)
    return train_loader, val_loader

def create_train_state(rng):
  cnn = CNN()
  variables=cnn.init(rng, jnp.ones([2,100, 65, 500,1]))
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
  schedule=create_learning_rate_fn(base_learning_rate=0.001,steps_per_epoch=steps_per_epoch)
  tx = optax.sgd(learning_rate=schedule,momentum= 0.90)
  state=train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
  return state,batch_stats

@jax.jit
def predict(state, batch_stats,image_i):
  logits= state.apply_fn({'params': state.params,"batch_stats":batch_stats},image_i,is_training=False)
  return logits
def test(state, batch_stats,test_ds):
  
  accuracy=0
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
  return accuracy/val_dataset.__len__()

def train_epoch(state, train_ds, batch_size, rng,batch_stats):
  train_ds_size = train_dataset.__len__()
  steps_per_epoch = train_ds_size // batch_size

  epoch_loss = []
  acc_count=0
  acc_total=0
  for i,(x,y) in enumerate(train_ds):
    batch_images = np.array(x)
    x1,x2=pywt.dwt(batch_images, 'haar')
    # print(x1.shape,x2.shape)
    x1=x1[...,None]
    x2=x2[...,None]
    batch_images=(x1,x2)
    batch_labels = np.array(y)
    grads, loss, accuracy ,batch_stats= apply_model(state, batch_images, batch_labels,batch_stats)
    
    state = update_model(state, grads)
    
    epoch_loss.append(loss)
    acc_count=0.1*accuracy+0.9*acc_count
    acc_total+=batch_labels.shape[0]
    if i%100==0:
      print(f"{acc_count:0.2f}",end=" ")
  train_loss = np.mean(epoch_loss)
  train_accuracy =acc_count
  return state, train_loss, train_accuracy,batch_stats
def save_weights(weights, filename):
    bytes_output = flax.serialization.to_bytes(weights)
    pickle.dump(bytes_output, open(filename, 'wb'))

def load_weights(weights, filename):
    pkl_file = pickle.load(open(filename, 'rb'))
    trained_wts = flax.serialization.from_bytes(weights, pkl_file)
    return trained_wts
def train_and_evaluate() -> train_state.TrainState:  
  rng = jax.random.PRNGKey(0)  
  rng, init_rng = jax.random.split(rng)
  state,batch_stats = create_train_state(init_rng)
  best=0.01
  filename="cnn_best2.npy"
  if os.path.exists(filename):
    weight={"state":state,"batch_stats":batch_stats,"p":best}
    weight=load_weights(weight, filename)
    state=weight["state"]
    batch_stats=weight["batch_stats"]
    best=weight["p"]
    print(f"load best={best}")
  for epoch in range(1, num_epochs + 1):
    train_ds, test_ds = get_datasets()
    rng, input_rng = jax.random.split(rng)
    begin=time.time()
    state, train_loss, train_accuracy,batch_stats = train_epoch(state, train_ds,
                                                    batch_size=batch_size,
                                                    rng=input_rng,batch_stats=batch_stats)    
    state=state.replace(params=state.params)
    print("")
    print(f"time:{time.time()-begin:0.2f}")
    p=test(state, batch_stats,test_ds)
    
    if p>best :  
      best=p
      print("save to ",filename)
      weight={"state":state,"batch_stats":batch_stats,"p":p}
      save_weights(weight, filename)
      

    print(f"test_acc={p:0.2f}",end=" ")
    print("")
    print("")
  
  return state

train_and_evaluate() 