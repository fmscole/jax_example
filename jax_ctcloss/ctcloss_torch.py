
import torch as np
import torch.nn.functional as F
np.cuda.set_device(0)
ninf =-1e30#-np.inf

def insert_blank(labels, blank=0):
    new_labels=[blank] 
    for l in labels:
        new_labels += [l, blank]
    return new_labels
def compute_loss(log_alpha,t,i,T,L):
    t_onehot=F.one_hot(t-1,num_classes=T)
    t_onehot=t_onehot.to(np.float32)
    log_alpha=np.einsum("btl,bt->bl",log_alpha,t_onehot)

    i1_onehot=F.one_hot(i-1,num_classes=L)
    i1_onehot=i1_onehot.to(np.float32)
    logprobs1=np.einsum("bl,bl->b",log_alpha,i1_onehot)

    i2_onehot=F.one_hot(i-2,num_classes=L)
    i2_onehot=i2_onehot.to(np.float32)
    logprobs2=np.einsum("bl,bl->b",log_alpha,i2_onehot)
    
    return np.logaddexp(logprobs1,logprobs2)
    
def alpha(log_y, labels,input_len,label_len):
    B,T, K = log_y.shape
    B,L=labels.shape

    one_hot=F.one_hot(labels,K)
    one_hot=one_hot.to(np.float32)
    log_y=log_y.to(np.float32)
    logprobs=np.einsum("blk,btk->tbl",one_hot,log_y)

    pre_log_alpha=np.ones((B,L))*ninf
    pre_log_alpha[:,0]=0.0
    pre_log_alpha=pre_log_alpha.to(device="cuda")
    
    mask=np.tensor(labels[:,:-2]==labels[:,2:],dtype=np.int32)
    mask=1-mask
    mask=F.pad(mask,(2,0,0,0))
    z=np.zeros_like(mask)
    z=z.to(np.float32)
    n=np.ones_like(mask)*ninf
    mask=np.where(mask>0,z,n)
    
    
    def loop_for_t(pre_log_alpha,t):
        a = pre_log_alpha 
        b = pre_log_alpha[:,:-1] 
        b=F.pad(b,(1,0,0,0),mode="constant",value=ninf)
        c=pre_log_alpha[:,:-2]  
        c=F.pad(c,(2,0,0,0),mode="constant",value=ninf)        
        d= np.logaddexp(a+t,b+t)   
        next_log_alpha= np.logaddexp(d,c+t+mask)
        return next_log_alpha

    t=[]
    for i in range(logprobs.size(0)):
        pre_log_alpha=loop_for_t(pre_log_alpha,logprobs[i])
        t.append(pre_log_alpha)
    next_log_alpha_t=np.stack(t,dim=0)   
    next_log_alpha_t=next_log_alpha_t.transpose(1,0)
    loss=compute_loss(next_log_alpha_t,input_len,label_len,T,L) 
    return -loss.mean()
def beta(log_y, labels):
    log_y=np.flip(log_y,dims=[1])
    return alpha(log_y, labels)

def ctcloss(logits, labels,input_len,label_len):
    '''
    logits:(B,T,K)
    labels:(B,L)
    input_len:(B,)
    label_len:(B,)
    '''
    log_y=F.log_softmax(logits,dim=-1) 
    blank_char=[insert_blank(i) for i in labels ]
   
    labels=np.tensor(blank_char, dtype=np.long)
    label_len=label_len*2+1  
    labels=labels.to(device="cuda")

    loss=alpha(log_y, labels,input_len,label_len)    
    

    # next_log_beta_t,_=beta(log_y, labels)  
    # next_log_beta_t=np.flip(next_log_beta_t,dims=[1])
    # alpha_beta=next_log_alpha_t+next_log_beta_t    
    # grad=alpha_beta-2*logprobs
        
    # loss_mean=np.mean(loss)
    # back=grad+loss_mean
    return loss

if __name__ =="__main__":
    import optax
    import numpy
    import time
    import jax
    n=127
    numpy.random.seed(0)
    logits=numpy.random.random((100,127,5990))
    logits=np.tensor(logits)
    input_len=np.tensor([n for i in range(100)])

    targets=numpy.random.randint(1,26,(100,20))
    targets=np.tensor(targets)
    target_len=np.tensor([20 for i in range(100)])
    

    logits=logits.to(device="cuda")
    targets=targets.to(device="cuda")
    input_len=input_len.to(device="cuda")
    target_len=target_len.to(device="cuda")
    losss_foo=np.jit.trace(ctcloss,(logits, targets,input_len,target_len))
    losss=losss_foo(logits, targets,input_len,target_len)
    print(losss)

    
    logits2=logits.transpose(1,0)
    start=time.time()
    for i in range(1000):
        losss=np.ctc_loss(logits2, targets,input_len,target_len)   
        print(losss[0],end=" ")
    print("")
    print("torch:")
    print(time.time()-start)



    start=time.time()
    for i in range(1000):
        losss=losss_foo(logits, targets,input_len,target_len)   
        print(losss,end=" ")
    print("")
    print("v1:")
    print(time.time()-start)
    