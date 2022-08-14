
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pandas import array
import tensorflow_datasets as tfds
import tensorflow as tf 

# from  stocks_dataset import x_train,x_test,y_train,y_test,data_min,data_max
from dataset_pytorch import training_generator,test_generator,trainset
batch_size=100
num_epochs=30
class SimpleScan(nn.Module):
    @nn.compact
    def __call__(self, xs,is_training:bool=True):
        LSTM = nn.scan(nn.LSTMCell,
                    variable_broadcast="params",
                    split_rngs={"params": False},
                    in_axes=1,
                    out_axes=1)    
        
       
        ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (xs.shape[0],), 100)
        ch, xs=LSTM()(ch, xs)
                
        xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)

        ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (xs.shape[0],), 100)
        ch, xs=LSTM()(ch, xs)        
        
        xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)
        ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (xs.shape[0],), 200)
        ch, xs=LSTM()(ch, xs)
        
        xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)
        ch = nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (xs.shape[0],), 200)
        ch, xs=LSTM()(ch, xs)
        xs=nn.Dropout(rate=0.2,deterministic=not is_training)(xs)
        # xs = nn.BatchNorm(use_running_average=not is_training)(xs)

        xs=xs[:,-1,:]
        xs=nn.Dense(features=6)(xs)        
        return xs

@jax.jit
def apply_model(state, images, labels,old_variables,dropout_rng):
  def loss_fn(params,old_variables):
    
    # logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_variables["batch_stats"]}, images,is_training=True, mutable=['batch_stats'],rngs={'dropout':dropout_rng})
    logits= state.apply_fn({'params': params}, images,is_training=True,rngs={'dropout':dropout_rng})
    mutated_vars =None
    loss = jnp.mean(jnp.sum(0.5*(logits-labels)**2,axis=-1))

    weight_penalty_params = jax.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (logits,mutated_vars)    

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits,new_batch_stats)), grads = grad_fn(state.params,old_variables)
  logits=logits*(trainset.data_max-trainset.data_min)+trainset.data_min
  labels=labels*(trainset.data_max-trainset.data_min)+trainset.data_min
  accuracy =jnp.sum(jnp.average(jnp.abs(logits-labels)/labels*100,axis=-1))  
  return grads, loss, accuracy,new_batch_stats,logits

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)



def create_train_state(rng):
  cnn = SimpleScan()
  key_1, key_2, key_3 = jax.random.split(jax.random.PRNGKey(0), 3)
  variables=cnn.init({'params': key_1, 'dropout':key_1}, jnp.ones([1, 60, 6]))
  params = variables['params']
  
  def create_learning_rate_fn(base_learning_rate: float,steps_per_epoch: int):
    warmup_epochs=5    
    warmup_fn = optax.linear_schedule(init_value=0.000001, end_value=base_learning_rate,transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,decay_steps=cosine_epochs * steps_per_epoch,alpha=0.0001)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn

  steps_per_epoch=60000//batch_size
  schedule=create_learning_rate_fn(base_learning_rate=0.01,steps_per_epoch=steps_per_epoch)
  # tx = optax.sgd(learning_rate=schedule,momentum= 0.90)
  tx = optax.adam(learning_rate=0.001)
  state=train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
  return state,variables

@jax.jit
def predict(state, variables,image_i):  
  # logits= state.apply_fn({'params': state.params,"batch_stats":variables["batch_stats"]},image_i,is_training=False)
  logits= state.apply_fn({'params': state.params},image_i,is_training=False)
  return logits
def accuracy(state, variables, test_ds):
  accuracy=0
  batch_size=50
  for (x,y) in test_generator:
    x=np.array(x)
    y=np.array(y)
    jax.device_put(x)
    jax.device_put(y)
    # print("1")
    # logits= predict(state, variables,x).block_until_ready()
    logits= predict(state, variables,x)
    # print("2")
    logits=logits*(trainset.data_max-trainset.data_min)+trainset.data_min
    label_i=y*(trainset.data_max-trainset.data_min)+trainset.data_min
    accuracy += jnp.sum(jnp.average(jnp.abs(logits-label_i)/label_i*100,axis=-1))  
  return accuracy/trainset.test_len

def train_epoch(state, train_ds, batch_size, rng,variables):
  
  epoch_loss = []
  sum_accuracy = 0
  for (x,y) in training_generator:
    x=np.array(x)
    y=np.array(y)
    
    jax.device_put(x)
    jax.device_put(y)
    rng, dropout_rng = jax.random.split(rng)
    grads, loss, accuracy ,variables,y= apply_model(state, x, y,variables,dropout_rng)
    
    state = update_model(state, grads)
    
    epoch_loss.append(loss)
    sum_accuracy +=accuracy
  train_loss = np.mean(epoch_loss)
  train_accuracy = sum_accuracy/trainset.train_len
  return state, train_loss, train_accuracy,variables,y

def train_and_evaluate() -> train_state.TrainState:
  train_ds, test_ds =None,None
  rng = jax.random.PRNGKey(0)  
  rng, init_rng = jax.random.split(rng)
  state,variables = create_train_state(init_rng)
  # print(jax.tree_util.tree_map(lambda x:x.shape,state.params))
  print(trainset.data_max,trainset.data_min)
  for epoch in range(1, 100 + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy,variables,y = train_epoch(state, train_ds,
                                                    batch_size=batch_size,
                                                    rng=input_rng,variables=variables)   
    print("") 
    print("epoch:",epoch)                                         
    print(f"train Error_ratio={train_accuracy:2.1f}%")     
    print(f"test Error_ratio={accuracy(state, variables, test_ds):2.1f}%")
    # print(y)
  
  return state
if __name__ == '__main__':
  train_and_evaluate() 