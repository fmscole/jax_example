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
from generator import data_set,data_loader
import pickle
import os
from crnn import CRNN
from config import cfg
# from ctcloss import CTCLoss

batch_size=100
num_epochs=100
class_nums=95
def remove_blank(labels, blank=0):
    new_labels = []
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    new_labels = [l for l in new_labels if l != blank]
    return new_labels

@jax.jit
def apply_model(state, batch,old_batch_stats):
  def loss_fn(params,old_batch_stats):
    images, target, input_len, target_len=batch
    
    images=images.transpose(0,2,3,1)
    logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_batch_stats}, images,is_training=True, mutable=['batch_stats'])
    
    label_paddings=jnp.where(target>0,0,1)
    logit_paddings=jnp.zeros(logits.shape[0:2])
    loss=optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=target,label_paddings=label_paddings)
    
    loss=jnp.mean(loss)

    # weight_penalty_params = jax.tree_leaves(params)
    # weight_decay = 0.0001
    # weight_l2 = sum(jnp.sum(x ** 2)
    #                  for x in weight_penalty_params
    #                  if x.ndim > 1)
    # weight_penalty = weight_decay * 0.5 * weight_l2
    # loss = loss + weight_penalty

    return loss, (logits,mutated_vars['batch_stats'])    
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits,new_batch_stats)), grads = grad_fn(state.params,old_batch_stats)
  return grads, loss,new_batch_stats,logits

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def create_train_state(rng):
  crnn = CRNN(class_nums=len(data_set.alpha))
  variables=crnn.init(rng, jnp.ones([100, 512, 32,1]))
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
  tx = optax.sgd(learning_rate=0.001,momentum= 0.90)
  state=train_state.TrainState.create(apply_fn=crnn.apply, params=params, tx=tx)
  return state,batch_stats

@jax.jit
def predict(state, batch_stats,image_i):
  logits= state.apply_fn({'params': state.params,"batch_stats":batch_stats},image_i,is_training=False)
  return logits
def test(state, batch_stats,batch,logits):  
  accuracy=0
  total=0
  image_i, target, input_len, target_len=batch
  # logits= predict(state, batch_stats,image_i)
  for i in range(logits.shape[0]):
    y=logits[i]
    y=np.argmax(y,axis=-1)
    y=remove_blank(y)
    label_len=target_len[i]
    y=np.array(y)
    label_len=np.minimum(y.shape[0],label_len)
    y=y[:label_len]
    label=target[i]
    label=label[:label_len]
    
    label=np.array(label)
    if y.shape!=label.shape:
      print(y.shape,label.shape)
    accuracy += np.sum(y == label)  
    total+=label_len
    # print(accuracy,y,label)
  return accuracy,label_len

def train_epoch(state, train_ds, batch_size, rng,batch_stats):
  train_ds_size = 60000
  steps_per_epoch = train_ds_size // batch_size

  epoch_loss = []
  acc_count=0
  acc_total=0
  for i,batch in enumerate(train_ds):
    
    grads, loss, batch_stats,logits= apply_model(state, batch,batch_stats)
    
    state = update_model(state, grads)
    
    epoch_loss.append(loss)
    
    print(loss,test(state, batch_stats,batch,logits),end=" ") 
    # acc_count=0.1*accuracy+0.9*acc_count
  #   acc_total+=batch_labels.shape[0]
  #   if i%100==0:
  #     print(f"{acc_count:0.2f}",end=" ")
  train_loss = np.mean(epoch_loss)
  # train_accuracy =acc_count
  return state, train_loss,batch_stats
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
  best=0
  filename="best.npy"
  if os.path.exists(filename):
    weight={"state":state,"batch_stats":batch_stats,"p":best}
    weight=load_weights(weight, filename)
    state=weight["state"]
    batch_stats=weight["batch_stats"]
    best=weight["p"]
    print(f"load best={best}")

  
  for epoch in range(1, num_epochs + 1):
    train_ds = data_loader
    rng, input_rng = jax.random.split(rng)
    begin=time.time()
    state, train_loss, batch_stats = train_epoch(state, train_ds,
                                                    batch_size=batch_size,
                                                    rng=input_rng,batch_stats=batch_stats)   
    print(train_loss) 
    # state=state.replace(params=state.params)
    # print("")
    # print(f"time:{time.time()-begin:0.2f}")
    # p=test(state, batch_stats,test_ds)
    
    # if p>best :  
    #   best=p
    #   print("save to ",filename)
    #   weight={"state":state,"batch_stats":batch_stats,"p":p}
    #   save_weights(weight, filename)
    #   weight={"state":state,"batch_stats":batch_stats,"p":p}
    #   weight=load_weights(weight, filename)
    #   state=weight["state"]
    #   batch_stats=weight["batch_stats"]
    #   p=weight["p"]
      

    # print(f"test_acc={p:0.2f}",end=" ")
    # print("")
    # print("")
  
  return state

train_and_evaluate() 