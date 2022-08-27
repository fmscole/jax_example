import jax.numpy as np
import jax

ninf =-1e31


def insert_blank(labels, blank=0):
    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels

def loop_for_i(st,t):
    target_len,pre_log_alpha,log_y,labels,one_hot,mask=st 
      
    a = pre_log_alpha 
    b = pre_log_alpha[:,:-1] 
    b=np.pad(b,((0,0),(1,0)),mode="constant",constant_values=(ninf,ninf))
    c=pre_log_alpha[:,:-2]  
    c=np.pad(c,((0,0),(2,0)),mode="constant",constant_values=(ninf,ninf))
    
    d= np.logaddexp(a,b)   
    e= np.logaddexp(d,c+mask)
    f=np.einsum("blk,bk->bl",one_hot,log_y[:,t])
    next_log_alpha=e+f
    return (target_len,next_log_alpha,log_y,labels,one_hot,mask),t

@jax.jit
def ctcloss(log_y, labels,target_len):
    log_y=jax.nn.log_softmax(log_y)
    target_len=target_len*2+1
    labels=np.array([insert_blank(list(i)) for i in labels ])
    B,T, V = log_y.shape
    labels=np.array(labels)
    one_hot=jax.nn.one_hot(labels,log_y.shape[-1])
    f=np.einsum("blk,bk->bl",one_hot,log_y[:,0])
    log_alpha=f[:,0:2]
    log_alpha=np.pad(log_alpha,((0,0),(0,f.shape[-1]-2)),mode="constant",constant_values=(ninf,ninf))
    
    tscan=np.array(range(1,T))

    
    mask=np.array(labels[:,:-2]==labels[:,2:],np.int32)
    mask=1-mask
    mask=np.pad(mask,((0,0),(2,0)))
    mask=np.where(mask>0,0,ninf)
    

    state=(target_len,log_alpha,log_y,labels,one_hot,mask)
    
    (target_len,log_alpha,log_y,labels,one_hot,mask),_=jax.lax.scan(loop_for_i,state,tscan)  
              
    return log_alpha[:,target_len-1]+log_alpha[:,target_len-2]
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
        print(losss[0],end="")
    print("")
    print(time.time()-start)

    
    logit_paddings=np.zeros(logits.shape[:2])
    label_paddings=np.where(targets>0,0.0,1.0)
    ctcloss2(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)

    start=time.time()
    for i in range(100):
        l=ctcloss2(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)
        print(l[0],end=" ")
    print("")
    print(time.time()-start)

    pass

    