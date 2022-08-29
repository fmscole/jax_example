import jax
import jax.numpy as jnp
def ctc_loss_with_forward_probs(logits,logit_paddings,labels,label_paddings,blank_id: int = 0,log_epsilon: float = -1e5):
    batchsize, unused_maxinputlen, num_classes = logits.shape
    batchsize_of_labels, maxlabellen = labels.shape

    logprobs = jax.nn.log_softmax(logits)
    labellens = maxlabellen - jnp.sum(label_paddings, axis=1).astype(jnp.int32)
    
    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    mask = (labels[:, :-1] == labels[:, 1:]).astype(jnp.float32)    #相邻重复mask ，比L少1   
    mask = jnp.pad(mask, ((0, 0), (0, 1)))  #右端pad，保持形长度相同
    
    logprobs_blank = logprobs[:, :, blank_id:blank_id + 1]  # [B, T, 1]    blank的概率值，为啥取这个名字
    logprobs_blank = jnp.transpose(logprobs_blank, (1, 0, 2))  # [T, B, 1]
    
    one_hot = jax.nn.one_hot(labels, num_classes=num_classes)  # [B, N, K]
    logprobs_char = jnp.einsum('btk,bnk->btn', logprobs, one_hot)   #非blank字符的概率值
    logprobs_char = jnp.transpose(logprobs_char, (1, 0, 2))  # [T, B, N]
    #在每一个时间片段，logprobs_phi，logprobs_emit都是常数

    logalpha_blank_init = jnp.ones(
        (batchsize, maxlabellen + 1)) * log_epsilon  # [B, N] #构造行向量，字符长度+1（blank）（先忽略batch维度），存储n+1个blank概率
    logalpha_blank_init = logalpha_blank_init.at[:, 0].set(0.0) #第一个设为0（log域为0，原值为1），因为循环体外初始化是虚拟的，目的是为了循环体内能够正确计算t=0时刻的值，
    
    logalpha_char_init = jnp.ones((batchsize, maxlabellen)) * log_epsilon #构造行向量，字符长度,存储字符概率
    
    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return jnp.concatenate(
            [phi[:, :1], jnp.logaddexp(phi[:, 1:], added_score)], axis=-1)

    def loop_body(prev, x):
        prev_blank, prev_char = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_blank_orig = prev_blank
        prev_blank = update_phi_score(prev_blank, prev_char + log_epsilon * mask)

        logprob_char, logprob_blank, pad = x

        # phi-to-emit transition
        next_char = jnp.logaddexp(prev_blank[:, :-1] + logprob_char,
                                prev_char + logprob_char)
        # self-loop transition
        next_blank = prev_blank + logprob_blank
        # emit-to-phi blank transition only when the next label is repetition
        next_blank = update_phi_score(
            next_blank, prev_char + logprob_blank + log_epsilon * (1.0 - mask))

        pad = pad.reshape((batchsize, 1))
        next_char = pad * prev_char + (1.0 - pad) * next_char
        next_blank = pad * prev_blank_orig + (1.0 - pad) * next_blank

        return (next_blank, next_char), (next_blank, next_char)

    xs = (logprobs_char, logprobs_blank, logit_paddings.transpose((1, 0)))
    _, (logalpha_phi,
        logalpha_emit) = jax.lax.scan(loop_body,
                                        (logalpha_blank_init, logalpha_char_init), xs)

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi = logalpha_phi.at[-1].set(logalpha_phi_last)

    # extract per_seq_loss
    one_hot = jax.nn.one_hot(labellens, num_classes=maxlabellen + 1)  # [B, N+1]
    per_seq_loss = -jnp.einsum('bn,bn->b', logalpha_phi_last, one_hot)

    return per_seq_loss, logalpha_phi, logalpha_emit


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