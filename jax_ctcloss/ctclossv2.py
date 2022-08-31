import jax.numpy as np
import jax

ninf =-1e5

def compute_loss(log_alpha_blank,log_alpha_char,t,i):
    return np.logaddexp(log_alpha_blank[t-1,i],log_alpha_char[t-1,i-1])

@jax.jit
def alpha(logits, labels,blank=0):
    log_y=jax.nn.log_softmax(logits)    

    B,T, K = log_y.shape
    B,L=labels.shape

    one_hot=jax.nn.one_hot(labels,K)
    logprobs_char=np.einsum("blk,btk->tbl",one_hot,log_y)
    logprobs_blank=log_y[:,:,blank:blank+1]
    logprobs_blank=logprobs_blank.transpose(1,0,2)

    pre_log_alpha_blank=np.ones((B,L+1))*ninf
    pre_log_alpha_blank=pre_log_alpha_blank.at[:,0].set(0.0)
    pre_log_alpha_char=np.ones((B,L))*ninf

    mask=np.array(labels[:,:-1]==labels[:,1:],np.int32)
    mask=1-mask
    mask=np.pad(mask,((0,0),(1,0)))
    mask=np.where(mask>0,0,ninf)
    
    def loop_for_t(pre_log_alpha,t):
        pre_log_alpha_blank,pre_log_alpha_char = pre_log_alpha 
        logprobs_blank,logprobs_char=t
        #blank
        next_log_alpha_blank_with_blank=pre_log_alpha_blank+logprobs_blank
        next_log_alpha_blank_with_char=pre_log_alpha_char+logprobs_blank
        next_log_alpha_blank_add=np.logaddexp(next_log_alpha_blank_with_blank[:,1:],next_log_alpha_blank_with_char)
        next_log_alpha_blank=np.concatenate([next_log_alpha_blank_with_blank[:,:1],next_log_alpha_blank_add],axis=-1)
        #label
        next_log_alpha_char_with_char=pre_log_alpha_char+logprobs_char
        next_log_alpha_char_with_blank=pre_log_alpha_blank[:,:-1] +logprobs_char
        add_blank_pretimechar=np.logaddexp(next_log_alpha_char_with_char,next_log_alpha_char_with_blank)
        char2 = pre_log_alpha_char[:,:-1] 
        char2=np.pad(char2,((0,0),(1,0)),mode="constant",constant_values=ninf)
        next_log_alpha_char=np.logaddexp(add_blank_pretimechar,char2+logprobs_char+mask)
        
        return (next_log_alpha_blank,next_log_alpha_char),(next_log_alpha_blank,next_log_alpha_char)

    _,(next_log_alpha_blank,next_log_alpha_char)=jax.lax.scan(loop_for_t,(pre_log_alpha_blank,pre_log_alpha_char),(logprobs_blank,logprobs_char))             
    return (next_log_alpha_blank,next_log_alpha_char) 
@jax.jit
def ctcloss(logits, labels,input_len,label_len):
    '''
    logits:(B,T,K)
    labels:(B,L)
    input_len:(B,)
    label_len:(B,)
    '''
    next_log_alpha_blank,next_log_alpha_char=alpha(logits, labels)  
    next_log_alpha_blank=next_log_alpha_blank.transpose(1,0,2)
    next_log_alpha_char=next_log_alpha_char.transpose(1,0,2)
    loss=jax.vmap(compute_loss,in_axes=0,out_axes=0)(next_log_alpha_blank,next_log_alpha_char,input_len,label_len)
    return -loss

if __name__ =="__main__":
    import optax
    import numpy
    import time
    n=127
    logits=numpy.random.random((100,127,5990))
    input_len=np.array([n for i in range(100)])

    targets=numpy.random.randint(1,3,(100,20))
    target_len=np.array([20 for i in range(100)])
    # targets=np.pad(targets,pad_width=((0,0),(0,60)))

    @jax.jit
    def ctcloss2(logits,logit_paddings,targets,label_paddings):
        return optax.ctc_loss(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)

    losss=ctcloss(logits, targets,input_len,target_len)
    print(losss)


    
    l=[0.0 for i in range(n)]+[1.0 for i in range(127-n)]
    logit_paddings=np.array([l for i in range(100)])
    label_paddings=np.where(targets>0,0.0,1.0)
    losss=ctcloss2(logits=logits,logit_paddings=logit_paddings,targets=targets,label_paddings=label_paddings)
    print(losss)

    # start=time.time()
    # for i in range(1000):
    #     l=ctcloss2(logits=logits,logit_paddings=logit_paddings,targets=targets,label_paddings=label_paddings)
    #     print(l[0],end=" ")
    # print("")
    # print("optax")
    # print(time.time()-start)
    

    # start=time.time()
    # for i in range(1000):
    #     losss=ctcloss(logits, targets,input_len,target_len)   
    #     print(losss[0],end=" ")
    # print("")
    # print("v2")
    # print(time.time()-start)