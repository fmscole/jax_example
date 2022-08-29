import jax.numpy as jnp
import jax
import functools

@jax.jit
def logadd(x0, x1, x2):
	# produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
	#return jax.nn.logsumexp(jnp.stack([x0, x1, x2]), axis=0)
	
	# use if -inf log-space zero element is used
	# return LogsumexpFunction.apply(x0, x1, x2)
	
	# produces inplace modification error https://github.com/pytorch/pytorch/issues/31819
	#x0 = x0.clone(); x1 = x1.clone(); x2 = x2.clone();
	m = jnp.max(jnp.stack([x0, x1, x2]))
	m = jnp.where(jnp.isinf(m), x=0, y=m)    
	res = jnp.exp(x0 - m) + jnp.exp(x1 - m) + jnp.exp(x2 - m)
	res = jnp.log(jnp.clip(res, a_min=1e-30))
	return res + m


@functools.partial(jax.jit, static_argnames=['dim'])
def gather_jax(tensor, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    if dim < 0:
        dim = len(tensor.shape) + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = tensor.shape[:dim] + tensor.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != jnp.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = jnp.swapaxes(tensor, 0, dim)
    index_swaped = jnp.swapaxes(index, 0, dim)
    gathered = jnp.choose(index_swaped, data_swaped, mode='clip')
    return jnp.swapaxes(gathered, 0, dim)

@functools.partial(jax.jit, static_argnames=['blank', 'zero_fp32', 'zero_fp16'])
def jax_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_fp32=jnp.float32('-inf'), zero_fp16=jnp.float16('-inf')):
	input_time_size, batch_size = log_probs.shape[:2]
	B = jnp.arange(batch_size)
	_t_a_r_g_e_t_s_ = jnp.concatenate([targets, targets[:, :1]], axis = -1)
	_t_a_r_g_e_t_s_ = jnp.stack([jnp.full_like(_t_a_r_g_e_t_s_, blank), _t_a_r_g_e_t_s_], axis = -1)
	_t_a_r_g_e_t_s_ = _t_a_r_g_e_t_s_.reshape(*_t_a_r_g_e_t_s_.shape[:-2], -1)
	diff_labels = jnp.concatenate([jnp.array(False).tile((batch_size, 2)), _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]], axis = 1)
	# print(diff_labels.shape)
	# if zero = float('-inf') is used as neutral element, custom logsumexp must be used to avoid nan grad in torch.logsumexp
	
	zero_padding, zero = 2, jnp.array(zero_fp16 if log_probs.dtype == jnp.float16 else zero_fp32, dtype = log_probs.dtype)
	log_probs_ = gather_jax(log_probs, -1, _t_a_r_g_e_t_s_[None, :, :].repeat(input_time_size, axis=0))
	log_alpha = jnp.full((input_time_size, batch_size, zero_padding + _t_a_r_g_e_t_s_.shape[-1]), zero, dtype=log_probs.dtype)
	log_alpha = jax.ops.index_update(log_alpha, jax.ops.index[0, :, zero_padding + 0], log_probs[0, :, blank])
	log_alpha = jax.ops.index_update(log_alpha, jax.ops.index[0, :, zero_padding + 1], log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]])
	# log_alpha[1:, :, zero_padding:] = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(len(log_probs), -1, -1))[1:]
	
	def ctc_loss_single_step(carry_state, current_x, diff_labels):
		current_log_probs, current_is_finished_mask = current_x
		# print(current_log_probs.shape, current_is_finished_mask.shape)
		prev_alpha_state = carry_state
		next_alpha_state = current_log_probs + logadd(prev_alpha_state[:, 2:], prev_alpha_state[:, 1:-1], jnp.where(diff_labels, prev_alpha_state[:, :-2], zero))
		next_alpha_state = jnp.concatenate([prev_alpha_state[:, :2], next_alpha_state], axis=1)
		# print('jnp.where shapes:', current_is_finished_mask.shape, prev_alpha_state.shape, next_alpha_state.shape)        
		next_alpha_state_finished_accounted = jnp.where(current_is_finished_mask[:, None], prev_alpha_state, next_alpha_state)
		return next_alpha_state_finished_accounted, None
	initial_log_alpha = jnp.full((batch_size, zero_padding + _t_a_r_g_e_t_s_.shape[-1]), zero, dtype=log_probs.dtype)
	initial_log_alpha = jax.ops.index_update(initial_log_alpha, jax.ops.index[:, zero_padding + 0], log_probs[0, :, blank])
	initial_log_alpha = jax.ops.index_update(initial_log_alpha, jax.ops.index[:, zero_padding + 1], log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]])
	
	is_finished = (jnp.arange(input_time_size)[None, :].repeat(batch_size, axis=0) >= input_lengths[:, None]).T
	# print(is_finished.shape, initial_log_alpha.shape, log_probs_.shape)
	# print(is_finished)
	initial_state = initial_log_alpha
	final_alpha, _ = jax.lax.scan(functools.partial(ctc_loss_single_step, diff_labels=diff_labels), initial_state, (log_probs_[1:], is_finished[1:]))
	# print(final_alpha.shape)
	
	# for t in range(1, input_time_size):
		# new_timestemp_result = log_probs_[t] + logadd(log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1], jnp.where(diff_labels, log_alpha[t - 1, :, :-2], zero))
		# log_alpha = jax.ops.index_update(log_alpha, jax.ops.index[t, :, 2:], new_timestemp_result)

	l1l2 = gather_jax(final_alpha, -1, jnp.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], axis = -1)) 
	loss = -jax.nn.logsumexp(l1l2, axis = -1)
	return loss

@functools.partial(jax.jit, static_argnames=['blank'])
def loss_and_grad(logits, targets, input_lengths, target_lengths, blank=0):
    loss_function = lambda logits: jax_ctc_loss(jax.nn.log_softmax(logits), targets, input_lengths, target_lengths, blank=blank).sum()
    loss = loss_function(logits)
    grad_function = jax.grad(loss_function)
    grad = grad_function(logits)
    return loss, grad