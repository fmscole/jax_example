import jax
import jax.numpy as jnp
def ctc_loss_with_forward_probs(logits,logit_paddings,labels,label_paddings,blank_id: int = 0,log_epsilon: float = -1e5):
    batchsize, unused_maxinputlen, num_classes = logits.shape
    batchsize_of_labels, maxlabellen = labels.shape

    log_probs = jax.nn.log_softmax(logits)
    labellens = maxlabellen - jnp.sum(label_paddings, axis=1).astype(jnp.int32)
    
    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    mask = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)    #相邻重复mask ，比L少1   
    mask = jnp.pad(mask, ((0, 0), (0, 1)))  #右端pad 1位，保持长度相同，mask的作用是实现错位相加
    
    log_probs_blank = log_probs[:, :, blank_id:blank_id + 1]  # [B, T, 1]    blank的概率值
    log_probs_blank = jnp.transpose(log_probs_blank, (1, 0, 2))  # [T, B, 1]  scan以T为导轴
    
    one_hot = jax.nn.one_hot(labels, num_classes=num_classes)  # [B, N, K]
    log_probs_char = jnp.einsum('btk,bnk->btn', log_probs, one_hot)   #非blank字符的概率值, 其实直接'btk,bnk->tbn'就可以了，不再需要transpose
    log_probs_char = jnp.transpose(log_probs_char, (1, 0, 2))  # [T, B, N],因为scan统一为axis=0为导轴
    #在每一个时间片段，logprobs_phi，logprobs_emit都是常数

    log_alpha_blank_init = jnp.ones(
        (batchsize, maxlabellen + 1)) * log_epsilon  # [B, N] #构造行向量，字符长度+1（blank）（先忽略batch维度），存储n+1个alpha值
    log_alpha_blank_init = log_alpha_blank_init.at[:, 0].set(0.0) #第一个设为0（log域为0，原值为1），因为循环体外初始化是虚拟的，目的是为了循环体内能够正确计算t=0时刻的值，
    
    log_alpha_char_init = jnp.ones((batchsize, maxlabellen)) * log_epsilon #构造行向量，字符长度,存储字符alpha
    
    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return jnp.concatenate(
            [phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1)

    def loop_body(prev, x):
        #每一个时间步，包括t=0
        prev_alpha_allblank, prev_alpha_char = prev # t-1时刻的alph值
        #需要更新，动态计算的两个量
        prev_blank_orig = prev_alpha_allblank
        #计算n+1个blank的alpha值（i-1）
        prev_alpha_allblank = update_phi_score(prev_alpha_allblank, prev_alpha_char + log_epsilon * mask)
        
        log_prob_char, log_prob_blank, logit_paddings = x

        # phi-to-emit transition
        #在原值计算中，先加后乘，在log域中反过来，先加再logaddexp，总之，减少复杂计算次数
        #先加t-1时刻的i和i-2的值
        #相邻字符错位相加
        next_char = jnp.logaddexp(prev_alpha_allblank[:, :-1] + log_prob_char,
                                prev_alpha_char + log_prob_char)
        # self-loop transition
        #blank实现上下
        next_blank = prev_alpha_allblank + log_prob_blank
        # emit-to-phi blank transition only when the next label is repetition
        #blank与char实现错位相加
        next_blank = update_phi_score(
            next_blank, prev_alpha_char + log_prob_blank + log_epsilon * (1.0 - mask))

        logit_paddings = logit_paddings.reshape((batchsize, 1))
        next_char = logit_paddings * prev_alpha_char + (1.0 - logit_paddings) * next_char
        next_blank = logit_paddings * prev_blank_orig + (1.0 - logit_paddings) * next_blank

        return (next_blank, next_char), (next_blank, next_char)

    xs = (log_probs_char, log_probs_blank, logit_paddings.transpose((1, 0)))
    _, (logalpha_phi,
        logalpha_emit) = jax.lax.scan(loop_body,
                                        (log_alpha_blank_init, log_alpha_char_init), xs)

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

    # extract per_seq_loss
    one_hot = jax.nn.one_hot(labellens, num_classes=maxlabellen + 1)  # [B, N+1]
    per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)

    return per_seq_loss


if __name__ =="__main__":
    import optax
    import numpy
    import time
    from jax_loss  import jax_ctc_loss
    logits=numpy.random.random((100,127,5990))

    targets=numpy.random.randint(1,26,(100,20))
    targets=numpy.pad(targets,pad_width=((0,0),(0,60)))
   
    logit_paddings=jnp.zeros(logits.shape[:2])
    label_paddings=jnp.where(targets>0,0.0,1.0)
    ctc_loss_with_forward_probs(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)

    # start=time.time()
    # for i in range(100):
    #     l=ctc_loss_with_forward_probs(logits=logits,logit_paddings=logit_paddings,labels=targets,label_paddings=label_paddings)
    #     print(l[0],end=" ")
    # print("")
    # print(time.time()-start)