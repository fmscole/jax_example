from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from functools import partial
from typing import Any, Callable, Sequence, Tuple
import time
from dataset import getDataLoader,numpy_collate

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
  """ResNetV1."""
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
    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
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
def apply_model(state,new_batch_stats, images, labels):
  def loss_fn(params):
    logits,mutable_vars = state.apply_fn({'params': params,"batch_stats":new_batch_stats}, images, mutable=['batch_stats'])
    one_hot = jax.nn.one_hot(labels, 1000)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    weight_penalty_params = jax.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (logits,mutable_vars['batch_stats'])
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits,old_batch_stats)), grads = grad_fn(state.params)
  y=jnp.argmax(logits, -1)
  accuracy = jnp.sum( y== labels)
  return grads, loss, accuracy,y,old_batch_stats

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def create_model(*, model_cls, **kwargs):
  model_dtype = jnp.float32
  return model_cls(num_classes=1000, dtype=model_dtype, **kwargs)
steps_per_epoch=40000//config.batch_size
def create_learning_rate_fn(config: ml_collections.ConfigDict,base_learning_rate: float,steps_per_epoch: int):
    warmup_fn = optax.linear_schedule(init_value=0.000001, end_value=base_learning_rate,transition_steps=config.warmup_epochs * steps_per_epoch)
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,decay_steps=cosine_epochs * steps_per_epoch,alpha=0.000001)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],boundaries=[config.warmup_epochs * steps_per_epoch])
    return schedule_fn
schedule=create_learning_rate_fn(config=config,base_learning_rate=0.1,steps_per_epoch=steps_per_epoch)

def create_train_state(rng):
  model=ResNet50(num_classes=1000)
  variables=model.init(rng, jnp.ones([config.batch_size, 224, 224, 3]))
  params = variables['params']
  batch_stats=variables["batch_stats"]  
  tx = optax.sgd(learning_rate=schedule,momentum=0.9)
  state=train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return state,batch_stats

@jax.jit
def predict(state,batch_stats,image_i):
  logits= state.apply_fn({'params': state.params,"batch_stats":batch_stats},image_i,train=False)
  return logits

def test(state,batch_stats,test_ds):
  accuracy=0
  total=0
  for i,( image_i,label_i) in enumerate(test_ds):
    logits= predict(state,batch_stats,image_i)
    total+=logits.shape[0]
    accuracy += jnp.sum(jnp.argmax(logits, -1) == label_i)  
  return accuracy/total

def train_epoch(state,batch_stats, train_ds):
  epoch_loss = []
  acc_count=0
  total=0
  for i,(batch_images,batch_labels) in enumerate(train_ds):    
    grads, loss, accuracy,y,batch_stats = apply_model(state,batch_stats, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    acc_count+=accuracy
    total+=batch_labels.shape[0]
  train_loss = np.mean(epoch_loss)
  train_accuracy = acc_count/total
  return state,batch_stats, train_loss, train_accuracy,y

def train_and_evaluate() -> train_state.TrainState:  
  train_ds=getDataLoader(batch_size=config.batch_size,is_train=True,collate_fn=numpy_collate)  
  print(config.batch_size)
  rng = jax.random.PRNGKey(0)  
  rng, init_rng = jax.random.split(rng)
  state,batch_stats = create_train_state(init_rng)
  
  for epoch in range(1, 200 + 1):
    test_ds=getDataLoader(batch_size=config.batch_size,is_train=False,collate_fn=numpy_collate)
    print("")
    print("epoch:",epoch)
    print("test_acc=",test(state,batch_stats, test_ds))

    begin=time.time()
    state,batch_stats, train_loss, train_accuracy,y = train_epoch(state,batch_stats, train_ds)    
    print("epoch_time:",time.time()-begin)
    print("lr:",schedule(epoch*40000//config.batch_size))
    print("train_loss:",train_loss)
    print("train_accuracy:",train_accuracy)
    print(y) 
  return state

train_and_evaluate() 