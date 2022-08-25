import jax.numpy as np
import jax

ninf =0 #-1e30

def _logsumexp(a, b):
    a,b=jax.lax.cond(a < b,lambda a,b:(b,a),lambda a,b:(a,b),a,b)    
    return a +b #np.log(1 + np.exp(b - a)) 

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
    
    # n=len(labels)
    # new_labels =np.zeros((2*n+1,)) 
    # state=(new_labels,labels,blank)
    # new_labels,labels,blank=jax.lax.fori_loop(0,n,loop_for_insert_blank,state)
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
    log_alpha=log_alpha.at[t, i].set( a * log_y[t, s])#
    return (t,log_alpha,log_y,labels),i
def loop_for_i(st,t):
    lscan,target_len,log_alpha,log_y,labels=st
    state=(t,log_alpha,log_y,labels)    
    (t,log_alpha,log_y,labels),_=jax.lax.scan(loop_for_fun,state,lscan)
    return (lscan,target_len,log_alpha,log_y,labels),t
def alpha(log_y, labels,target_len):
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
    return log_alpha[-1,labels[target_len]-1]+log_alpha[-1,labels[target_len]-2],log_alpha
    return log_alpha,log_y
@jax.jit
def ctcloss(logits, targets,target_len):
    return jax.vmap(alpha, in_axes=(0), out_axes=0)(logits, targets,target_len)

if __name__ =="__main__":
    import optax
    import numpy
    from jax_loss  import jax_ctc_loss
    # logits=numpy.random.random((1,127,5990))
    
    logits=numpy.ones((1,8,26))
    # logits=jax.nn.softmax(logits)
    logits=np.array([[0.24654511, 0.18837589 ,0.16937668 ,0.16757465, 0.22812766],
            [0.25443629, 0.14992236 ,0.22945293, 0.17240658, 0.19378184],
            [0.24134404 ,0.17179604 ,0.23572466, 0.12994237 ,0.22119288],
            [0.27216255 ,0.13054313, 0.2679252,  0.14184499 ,0.18752413],
            [0.32558002 ,0.13485564 ,0.25228604, 0.09743785, 0.18984045],
            [0.23855586, 0.14800386 ,0.23100255, 0.17158135, 0.21085638],
            [0.38534786 ,0.11524603, 0.18220093, 0.14617864, 0.17102655],
            [0.21867406 ,0.18511892, 0.21305488, 0.16472572, 0.21842642],
            [0.29856607 ,0.13646801, 0.27196606, 0.11562552, 0.17737434],
            [0.242347  , 0.14102063, 0.21716951, 0.2355229,  0.16393996],
            [0.26597326 ,0.10009752 ,0.23362892 ,0.24560198, 0.15469832],
            [0.23337289 ,0.11918746 ,0.28540761, 0.20197928 ,0.16005275]])
    # targets=numpy.random.randint(1,26,(1,20))
    targets=numpy.array([3,3,4])
    # targets=numpy.pad(targets,pad_width=((0,0),(0,6)))
    print(targets)
    target_len=numpy.array(3)
    losss=alpha(logits, targets,target_len)
    print(losss[0])
    print(losss[1])
    print(losss[2])
    # logit_paddings=np.zeros(logits.shape[:2])
    # label_paddings=np.where(targets>0,0.0,1.0)
    # # print(label_paddings)
    # print(optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings))
    # jax_ctc_loss(logits[0], targets[0], input_lengths=9, target_lengths=4, blank=0)

    pass

    