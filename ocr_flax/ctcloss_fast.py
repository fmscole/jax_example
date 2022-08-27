import jax.numpy as np
import jax

ninf =-1e30

def _logsumexp(a, b):
    a,b=jax.lax.cond(a < b,lambda a,b:(b,a),lambda a,b:(a,b),a,b)    
    return a +np.log(1 + np.exp(b - a)) 
def logsumexpv(a,b):
    return jax.vmap(_logsumexp)(a,b)


def insert_blank(labels, blank=0):
    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels

def loop_for_i(st,t):
    lscan,target_len,log_alpha,log_y,labels,one_hot,mask=st       
    a = log_alpha[t-1, :] 
    b = log_alpha[t-1, :-1] 
    b=np.pad(b,(1,0),mode="constant",constant_values=(ninf,ninf))
    c=log_alpha[t-1, :-2]  
    c=np.pad(c,(2,0),mode="constant",constant_values=(ninf,ninf))

    d= logsumexpv(a,b)   
    e= logsumexpv(d,c+mask)
    f=np.dot(one_hot,log_y[t])

    log_alpha=log_alpha.at[t].set(e+f)
    return (lscan,target_len,log_alpha,log_y,labels,one_hot,mask),t
def alpha(log_y, labels,target_len):
    log_y=jax.nn.log_softmax(log_y)
    target_len=target_len*2+1
    labels=np.array(insert_blank(list(labels)))
    T, V = log_y.shape
    L = len(labels)
    log_alpha = np.ones([T, L]) * ninf
    log_alpha=log_alpha.at[0, 0].set(log_y[0, labels[0]])
    log_alpha=log_alpha.at[0, 1] .set(log_y[0, labels[1]])
    lscan=np.array(range(L))
    tscan=np.array(range(1,T))

    labels=np.array(labels)
    mask=np.array(labels[:-2]==labels[2:],np.int32)
    mask=1-mask
    mask=np.pad(mask,(2,0))
    mask=np.where(mask>0,0,ninf)
    one_hot=jax.nn.one_hot(labels,log_y.shape[-1])

    state=(lscan,target_len,log_alpha,log_y,labels,one_hot,mask)
    
    (lscan,target_len,log_alpha,log_y,labels,one_hot,mask),_=jax.lax.scan(loop_for_i,state,tscan)            
    # return log_alpha[-1,target_len-1]+log_alpha[-1,target_len-2]
    return log_alpha[-1,target_len-1]+log_alpha[-1,target_len-2]
@jax.jit
def ctcloss(logits, targets,target_len):
    return jax.vmap(alpha, in_axes=(0), out_axes=0)(logits, targets,target_len)

if __name__ =="__main__":
    import optax
    import numpy
    import time
    from jax_loss  import jax_ctc_loss
    # logits=numpy.random.random((1,127,5990))
    # logits=jax.nn.softmax(logits)
    logits=numpy.ones((1,20,26))
    # logits=np.exp(logits)
    # logits=jax.nn.softmax(logits)

    # targets=numpy.random.randint(1,26,(1,20))
    targets=numpy.array([[1,2,2,2,2]])
    # targets=numpy.pad(targets,pad_width=((0,0),(0,6)))
    print(targets)
    target_len=numpy.array([5])
    start=time.time()
    for i in range(1000):
        losss=ctcloss(logits, targets,target_len)    
        print(losss,end="")

    print(time.time()-start)

    
    logit_paddings=np.zeros(logits.shape[:2])
    label_paddings=np.where(targets>0,0.0,1.0)

    # start=time.time()
    # for i in range(100):
    #     print(optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings),end="")
    # print(time.time()-start)

    pass

    