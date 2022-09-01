import jax
import jax.numpy as jnp

import numpy 
import time

import jax.numpy as jnp
from jax import jit, grad

import datasets

train_data,train_label,test_data,test_label=datasets.mnist(permute_train=True)
print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)

def test(params,test_data):   
    output = predict(params,test_data)
    p=jnp.mean(jnp.argmax(output,axis=1)==jnp.argmax(test_label,axis=1))
    print ( p,end="  ")
def activation_function1(x):
    return jnp.where(x>0,x,0)  
def activation_function2(x):
    x-=jnp.max(x)    
    return jnp.exp(x)/jnp.sum(jnp.exp(x))
def predict(param,inputs_list):
    hidden_inputs = jnp.dot( inputs_list,param[0])
    hidden_outputs = activation_function1(hidden_inputs)
    
    final_inputs = jnp.dot(hidden_outputs,param[1])
    final_outputs = activation_function2(final_inputs)
    
    return final_outputs
def loss(param,batch):  
    inputs_list, targets_list=batch
    final_outputs=predict(param,inputs_list)  
    t= -targets_list *jnp.log(final_outputs)  
    loss=jnp.sum(t)    
    return  loss

backward=jit(grad(loss))

lr=0.001
batchsize=100
moments=0.9

wih = jnp.array(numpy.random.normal(0.0, pow(784, -0.5), (784,400)))
who = jnp.array(numpy.random.normal(0.0, pow(784, -0.5), (400,10)))
params=[wih,who]

updates=jax.tree_util.tree_map(lambda x:jnp.zeros_like(x),params)

for j in range(100):
    for i in range(0,train_data.shape[0],batchsize):
            inputs=train_data[i:i+batchsize]
            targets=train_label[i:i+batchsize]
            batch=[inputs,targets]
            g=backward(params,batch)

            #SDG
            # params[0]=params[0]-lr*g[0]
            # params[1]=params[1]-lr*g[1]

            #DSG+Moment
            updates=jax.tree_util.tree_map(lambda g,updates:lr*g+moments*updates,g,updates) 
            params=jax.tree_util.tree_map(lambda params,updates: params-updates,params,updates)


    test(params,test_data)