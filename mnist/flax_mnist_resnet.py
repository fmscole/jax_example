
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf 
import time
from functools import partial
from typing import Any, Callable, Sequence, Tuple    
ModuleDef = Any

batch_size=100
num_epochs=100
# class CNN(nn.Module):
#   @nn.compact
#   def __call__(self, x,is_training:bool=True):
#     x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#     x = nn.BatchNorm(use_running_average=not is_training)(x)
#     x = nn.relu(x)
#     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#     x = nn.BatchNorm(use_running_average=not is_training)(x)
#     x = nn.relu(x)
#     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = x.reshape((x.shape[0], -1))  # flatten
#     x = nn.Dense(features=256)(x)
#     x = nn.relu(x)
#     x = nn.Dense(features=10)(x)
#     return x

ModuleDef = Any
class ResNetBlock(nn.Module):
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)
    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)
    return self.act(residual + y)

class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  @nn.compact
  def __call__(self, x):
    residual = x
    # print(x.shape)
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)
    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)
    return self.act(residual + y)

class ResNet(nn.Module):
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)
    x = conv(self.num_filters, (3, 3), (1, 1),
             padding='SAME',
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    # x = nn.max_pool(x, (3, 3), strides=(1, 1), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if (i==2 or i==3) and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x

ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)

@jax.jit
def apply_model(state, images, labels,old_batch_stats):
  def loss_fn(params,old_batch_stats):
    logits,mutated_vars = state.apply_fn({'params': params,"batch_stats":old_batch_stats}, images,train=True, mutable=['batch_stats'])
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



def create_train_state(rng):
#   cnn = CNN()
  cnn=ResNet18(num_classes=10)
  variables=cnn.init(rng, jnp.ones([batch_size, 28, 28, 1]))
  params = variables['params']
  
  batch_stats=variables['batch_stats']

  
  def create_learning_rate_fn(base_learning_rate: float,steps_per_epoch: int):
    warmup_epochs=5    
    warmup_fn = optax.linear_schedule(init_value=0.000001, end_value=base_learning_rate,transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,decay_steps=cosine_epochs * steps_per_epoch,alpha=0.000001)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn

  steps_per_epoch=60000//batch_size
  schedule=create_learning_rate_fn(base_learning_rate=0.1,steps_per_epoch=steps_per_epoch)
  tx = optax.sgd(learning_rate=schedule,momentum= 0.90)
  state=train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
  return state,batch_stats

@jax.jit
def predict(state, batch_stats,image_i):
  logits= state.apply_fn({'params': state.params,"batch_stats":batch_stats},image_i,train=False)
  return logits
def test(state, batch_stats,test_ds):
  images = test_ds['image']
  labels = test_ds['label']
  batchs=batch_size
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

def get_datasets():
  with tf.device('/cpu:0'):
    ds_builder = tfds.builder('fashion_mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds
    

def train_and_evaluate() -> train_state.TrainState:
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)  
  rng, init_rng = jax.random.split(rng)
  state,batch_stats = create_train_state(init_rng)
  
  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    
    print("")
    print("epoch:",epoch)
    print("test_acc=",test(state, batch_stats,test_ds))

    begin =time.time()
    state, train_loss, train_accuracy,batch_stats = train_epoch(state, train_ds,
                                                    batch_size=batch_size,
                                                    rng=input_rng,batch_stats=batch_stats)    
    
    print("time:",time.time()-begin)
  
  return state

train_and_evaluate() 