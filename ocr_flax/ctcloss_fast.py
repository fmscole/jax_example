import jax.numpy as np
import jax

ninf =-1e30

def insert_blank(labels, blank=0):
    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels

@jax.jit
def ctcloss(log_y, labels,target_len):
    log_y=jax.nn.log_softmax(log_y)    
    labels=np.array([insert_blank(i) for i in labels ])

    B,T, K = log_y.shape
    B,L=labels.shape

    one_hot=jax.nn.one_hot(labels,K)
    logprobs=np.einsum("blk,btk->tbl",one_hot,log_y)

    pre_log_alpha=np.ones((B,L))*ninf
    pre_log_alpha=pre_log_alpha.at[:,0].set(0.0)
    
    mask=np.array(labels[:,:-2]==labels[:,2:],np.int32)
    mask=1-mask
    mask=np.pad(mask,((0,0),(2,0)))
    mask=np.where(mask>0,0,ninf)
    
    def loop_for_t(pre_log_alpha,t):
        a = pre_log_alpha 
        b = pre_log_alpha[:,:-1] 
        b=np.pad(b,((0,0),(1,0)),mode="constant",constant_values=ninf)
        c=pre_log_alpha[:,:-2]  
        c=np.pad(c,((0,0),(2,0)),mode="constant",constant_values=ninf)
        
        d= np.logaddexp(a,b)   
        e= np.logaddexp(d,c+mask)
        next_log_alpha=e+t
        return next_log_alpha,None

    next_log_alpha,_=jax.lax.scan(loop_for_t,pre_log_alpha,logprobs)  

    target_len=target_len*2+1
    loss=jax.vmap(lambda log_alpha,i :np.logaddexp(log_alpha[i-1],log_alpha[i-2]))(next_log_alpha,target_len)       
    return -loss

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
    