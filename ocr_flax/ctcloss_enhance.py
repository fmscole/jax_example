import jax.numpy as np
import jax

ninf =-1e30


def insert_blank(labels, blank=0):
    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels

def loop_for_i(st,t):
    log_alpha,log_y,one_hot,mask=st       
    a = log_alpha[t-1, :] 
    b = log_alpha[t-1, :-1] 
    b=np.pad(b,(1,0),mode="constant",constant_values=(ninf,ninf))
    c=log_alpha[t-1, :-2]  
    c=np.pad(c,(2,0),mode="constant",constant_values=(ninf,ninf))

    d= np.logaddexp(a,b)   
    e= np.logaddexp(d,c+mask)
    f=np.dot(one_hot,log_y[t])

    log_alpha=log_alpha.at[t].set(e+f)
    return (log_alpha,log_y,one_hot,mask),t
def alpha(log_y, labels,target_len):
    log_y=jax.nn.log_softmax(log_y)
    
    labels=np.array(insert_blank(list(labels)))
    T, V = log_y.shape
    L = len(labels)
    log_alpha = np.ones([T, L]) * ninf
    log_alpha=log_alpha.at[0, 0].set(log_y[0, labels[0]])
    log_alpha=log_alpha.at[0, 1] .set(log_y[0, labels[1]])
    tscan=np.array(range(1,T))

    labels=np.array(labels)
    mask=np.array(labels[:-2]==labels[2:],np.int32)
    mask=1-mask
    mask=np.pad(mask,(2,0))
    mask=np.where(mask>0,0,ninf)
    one_hot=jax.nn.one_hot(labels,log_y.shape[-1])

    state=(log_alpha,log_y,one_hot,mask)
    
    (log_alpha,log_y,one_hot,mask),_=jax.lax.scan(loop_for_i,state,tscan) 

    target_len=target_len*2+1
    return np.logaddexp(log_alpha[-1,target_len-1],log_alpha[-1,target_len-2])
@jax.jit
def ctcloss(logits, targets,target_len):
    return jax.vmap(alpha, in_axes=(0), out_axes=0)(logits, targets,target_len)

@jax.jit
def ctcloss2(logits,logit_paddings,targets,label_paddings):
    return optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)

if __name__ =="__main__":
    import optax
    import numpy
    import time
    from jax_loss  import jax_ctc_loss
    logits=numpy.random.random((100,127,5990))

    targets=numpy.random.randint(1,26,(100,20))
    targets=np.array([insert_blank(list(i)) for i in targets ])
    targets=numpy.pad(targets,pad_width=((0,0),(0,60)))

    target_len=numpy.array([20 for i in range(100)])

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

    