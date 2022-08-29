import jax.numpy as np
import jax

ninf =-1e30

def _logsumexp(a, b):
    a,b=jax.lax.cond(a < b,lambda a,b:(b,a),lambda a,b:(a,b),a,b)    
    return a +np.log(1 + np.exp(b - a)) 

# def logsumexp(*args):
#     res = args[0]
#     for e in args[1:]:
#         res = _logsumexp(res, e)
#     return res
def loop_for_insert_blank(i,state):
    new_labels,labels,blank=state
    new_labels.at[2*i+1].set(labels[i])
    return (new_labels,labels,blank)

def insert_blank(labels, blank=0):

    new_labels=[blank]
    for l in labels:
        new_labels += [l, blank]
    new_labels=np.array(new_labels)
    return new_labels
def loop_for_fun(state,i):
    t,log_alpha,log_y,labels=state
    s = labels[i]
    a = log_alpha[t - 1, i]
    a=jax.lax.cond(i>=1,lambda a:_logsumexp(a, log_alpha[t - 1, i - 1]),lambda a: a,a)
    a=jax.lax.cond((i>=2) & (s!=0) & (s!=labels[i - 2]) ,lambda a:_logsumexp(a, log_alpha[t - 1, i - 2]),lambda a: a,a)
    log_alpha=log_alpha.at[t, i].set( a + log_y[t, s])#
    return (t,log_alpha,log_y,labels),i
def loop_for_i(st,t):
    lscan,target_len,log_alpha,log_y,labels=st
    state=(t,log_alpha,log_y,labels)    
    (t,log_alpha,log_y,labels),_=jax.lax.scan(loop_for_fun,state,lscan)
    return (lscan,target_len,log_alpha,log_y,labels),t
def alpha(log_y, labels,target_len):
    log_y=jax.nn.log_softmax(log_y)
    target_len=target_len*2+1
    labels=np.array(insert_blank(labels))
    T, V = log_y.shape
    L = len(labels)
    log_alpha = np.ones([T, L]) * ninf
    log_alpha=log_alpha.at[0, 0].set(log_y[0, labels[0]])
    log_alpha=log_alpha.at[0, 1] .set(log_y[0, labels[1]])
    lscan=np.array(range(L))
    tscan=np.array(range(1,T))
    state=(lscan,target_len,log_alpha,log_y,labels)
    
    (lscan,target_len,log_alpha,log_y,labels),_=jax.lax.scan(loop_for_i,state,tscan)            
    return np.logaddexp(log_alpha[-1,target_len-1],log_alpha[-1,target_len-2])
@jax.jit
def ctcloss(logits, targets,target_len):
    return -jax.vmap(alpha, in_axes=(0), out_axes=0)(logits, targets,target_len)

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

    start=time.time()
    for i in range(100):
        losss=ctcloss(logits, targets,target_len)    
        print(losss[0],end="")

    print(time.time()-start)

    
    # logit_paddings=np.zeros(logits.shape[:2])
    # label_paddings=np.where(targets>0,0.0,1.0)
    # start=time.time()
    # for i in range(100):
    #     print(optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings),end="")
    # print(time.time()-start)

    pass



    