
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf 

batch_size=100
num_epochs=30
class CNN(nn.Module):
  @nn.compact
  def __call__(self, x,is_training:bool=True):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.BatchNorm(use_running_average=not is_training)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    
    # x1=x
    # q = nn.Dense(features=32)(x1)    
    # k = nn.Dense(features=32)(x1)
    # v = nn.Dense(features=32)(x1)
    # bef=jnp.einsum("bmnk,bmjk->bnj",q,k)/8
    # x3=nn.softmax(bef,axis=-1)
    # x3=x3[...,None]
    # aft=jnp.einsum("bmnk,bnjk->bmj",v,x3)
    # x=aft

    # print(x.shape)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(features=10)(x)
    return x

@jax.jit
def apply_model(state, images, labels,old_batch_stats):
  def loss_fn(params,old_batch_stats):
    logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_batch_stats}, images,is_training=True, mutable=['batch_stats'])
    one_hot = jax.nn.one_hot(labels, 10)
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
  with tf.device('/cpu:0'):
    ds_builder = tfds.builder('fashion_mnist')
    # ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds

def create_train_state(rng):
  cnn = CNN()
  variables=cnn.init(rng, jnp.ones([100, 28, 28, 1]))
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
def test(state, batch_stats,test_ds):
  images = test_ds['image']
  labels = test_ds['label']
  batchs=100
  accuracy=0
  for i in range(0,len(images),batchs):
    image_i=images[i:i+batchs]
    label_i=labels[i:i+batchs]
    logits= predict(state, batch_stats,image_i)
    accuracy += jnp.sum(jnp.argmax(logits, -1) == label_i)  
  return accuracy/len(images)

def train_epoch(state, train_ds, batch_size, rng,batch_stats):
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []
  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy ,batch_stats= apply_model(state, batch_images, batch_labels,batch_stats)
    
    state = update_model(state, grads)
    
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy,batch_stats

def train_and_evaluate() -> train_state.TrainState:
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)  
  rng, init_rng = jax.random.split(rng)
  state,batch_stats = create_train_state(init_rng)
  
  for epoch in range(1, 100 + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy,batch_stats = train_epoch(state, train_ds,
                                                    batch_size=batch_size,
                                                    rng=input_rng,batch_stats=batch_stats)    
    
    print(test(state, batch_stats,test_ds),end=" ")
  
  return state

train_and_evaluate() 