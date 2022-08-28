import jax.numpy as np
import jax

ninf =-1e31

def _logsumexp(a, b):
    a,b=jax.lax.cond(a < b,lambda a,b:(b,a),lambda a,b:(a,b),a,b)    
    return a +np.log(1 + np.exp(b - a)) 
# def logsumexpv(a,b):
#     a_=[[j for j in i] for i in a]
#     b_=[[j for j in i] for i in b]
#     c=jax.tree_util.tree_map(_logsumexp,a_,b_)
#     c=np.array(c)
#     return c
def logsumexpv(a,b):
    return jax.vmap(_logsumexp)(a,b)
def vlogsumexp(a,b):
    return jax.vmap(logsumexpv)(a,b)

def insert_blank(labels, blank=0):
    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels
def compute_loss(log_alpha,i):
    return np.logaddexp(log_alpha[i-1],log_alpha[i-2])
def loop_for_i(st,t):
    pre_log_alpha,log_y,one_hot,mask=st 
      
    a = pre_log_alpha 
    b = pre_log_alpha[:,:-1] 
    b=np.pad(b,((0,0),(1,0)),mode="constant",constant_values=ninf)
    c=pre_log_alpha[:,:-2]  
    c=np.pad(c,((0,0),(2,0)),mode="constant",constant_values=ninf)
    
    d= np.logaddexp(a,b)   
    e= np.logaddexp(d,c+mask)
    f=np.einsum("blk,bk->bl",one_hot,log_y[:,t])
    next_log_alpha=e+f
    return (next_log_alpha,log_y,one_hot,mask),t

@jax.jit
def ctcloss(log_y, labels,target_len):
    log_y=jax.nn.log_softmax(log_y)
    
    labels=np.array([insert_blank(list(i)) for i in labels ])

    B,T, K = log_y.shape
    B,L=labels.shape
    one_hot=jax.nn.one_hot(labels,K)
    t0=np.einsum("blk,bk->bl",one_hot,log_y[:,0])
    pre_log_alpha=t0[:,0:2]
    pre_log_alpha=np.pad(pre_log_alpha,((0,0),(0,L-2)),mode="constant",constant_values=ninf)
    
    tscan=np.array(range(1,T))

    
    mask=np.array(labels[:,:-2]==labels[:,2:],np.int32)
    mask=1-mask
    mask=np.pad(mask,((0,0),(2,0)))
    mask=np.where(mask>0,0,ninf)
    
    
    state=(pre_log_alpha,log_y,one_hot,mask)
    
    (next_log_alpha,log_y,one_hot,mask),_=jax.lax.scan(loop_for_i,state,tscan)  

    target_len=target_len*2+1
    loss=jax.vmap(compute_loss)(next_log_alpha,target_len)
       
    return loss
    return log_alpha

@jax.jit
def ctcloss2(logits,logit_paddings,targets,label_paddings):
    return optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)

if __name__ =="__main__":
    import optax
    import numpy
    import time
    from jax_loss  import jax_ctc_loss
    logits=numpy.random.random((100,127,5990))
    # logits=jax.nn.softmax(logits)
    # logits=numpy.ones((1,20,26))
    # logits=np.exp(logits)
    # logits=jax.nn.softmax(logits)

    targets=numpy.random.randint(1,26,(100,20))
    # targets=numpy.array([[1,2,2,2,2]])
    targets=np.array([insert_blank(list(i)) for i in targets ])
    targets=numpy.pad(targets,pad_width=((0,0),(0,60)))
    # print(labels)
    target_len=numpy.array([[20] for i in range(100)])

    losss=ctcloss(logits, targets,target_len)
    
    start=time.time()
    for i in range(100):
        losss=ctcloss(logits, targets,target_len)   
        print(losss[0],end=" ")
    print("")
    print(time.time()-start)

    
    logit_paddings=np.zeros(logits.shape[:2])
    label_paddings=np.where(targets>0,0.0,1.0)
    ctcloss2(logits=logits,logit_paddings=logit_paddings,targets=targets,label_paddings=label_paddings)

    start=time.time()
    for i in range(100):
        l=ctcloss2(logits=logits,logit_paddings=logit_paddings,targets=targets,label_paddings=label_paddings)
        print(l[0],end=" ")
    print("")
    print(time.time()-start)

    pass

    