"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    token_to_id = {token: index for index, token in enumerate(specials)}

    for sentence in sentences:
        for token in sentence.split():
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    return token_to_id

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    return {token_id: token for token, token_id in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    return [token_to_id.get(token, token_to_id[unk_token]) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    return [id_to_token[token_id] for token_id in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
        return ids[:max_len] + [pad_id] * max(0, max_len - len(ids))

# Step 6 - stack_padded_sequences_to_batch
import torch
def stack_padded_sequences_to_batch(padded_sequences): return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
def scale_embeddings_by_sqrt_d_model(embeddings, d_model): return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import math
import torch
def compute_positional_div_term(d_model):
    return torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

# Step 9 - build_position_index_column
import torch
def build_position_index_column(max_len):
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    pe[:, 0::2] = torch.sin(position * div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    return torch.stack((pe[:, 0::2], torch.cos(position * div_term)), dim=-1).reshape_as(pe)

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    return torch.stack((torch.sin(torch.arange(max_len, dtype=torch.float32).unsqueeze(1) * torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))), torch.cos(torch.arange(max_len, dtype=torch.float32).unsqueeze(1) * torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)))), dim=-1).reshape(max_len, d_model)

# Step 13 - add_positional_encoding_to_embeddings
def add_positional_encoding_to_embeddings(embeddings, positional_encoding):
        return embeddings + positional_encoding[:embeddings.size(1)].unsqueeze(0)

# Step 14 - build_padding_mask
def build_padding_mask(token_ids, pad_id):
        return (token_ids != pad_id).unsqueeze(1).unsqueeze(1)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
def combine_padding_and_causal_masks(padding_mask, causal_mask):
        return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
def compute_raw_attention_scores(query, key):
        return query @ key.transpose(-2, -1)

# Step 18 - scale_attention_scores
def scale_attention_scores(scores, d_k):
        return scores / (d_k ** 0.5)

# Step 19 - mask_attention_scores_with_neg_inf
def mask_attention_scores_with_neg_inf(scores, mask):
        return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    return torch.where(torch.isneginf(masked_scores).all(dim=-1, keepdim=True), torch.zeros_like(masked_scores), torch.softmax(masked_scores, dim=-1))

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch
def scaled_dot_product_attention(query, key, value, mask=None):
    return (lambda scores: (lambda weights: (weights @ value, weights))(torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)))(((query @ key.transpose(-2, -1)) / (query.size(-1) ** 0.5)).masked_fill(~mask, float("-inf")) if mask is not None else (query @ key.transpose(-2, -1)) / (query.size(-1) ** 0.5))

# Step 23 - split_last_dim_into_heads
import torch
def split_last_dim_into_heads(tensor, num_heads):
    B, L, d_model = tensor.shape
    d_k = d_model // num_heads
    return tensor.reshape(B, L, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch
def transpose_heads_before_sequence(tensor):
    return tensor.permute(0, 2, 1, 3)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    return multi_head_tensor.transpose(1, 2).contiguous().view(multi_head_tensor.shape[0], multi_head_tensor.shape[2], -1)

# Step 26 - apply_linear_projection
import torch

def apply_linear_projection(x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)

# Step 27 - project_to_query_key_value
import torch

def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    return (torch.nn.functional.linear(x, w_q, b_q), torch.nn.functional.linear(x, w_k, b_k), torch.nn.functional.linear(x, w_v, b_v))

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    return tuple(t.view(t.shape[0], t.shape[1], num_heads, -1).transpose(1, 2) for t in (q, k, v))

# Step 29 - multi_head_scaled_dot_product_attention
def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
def merge_heads_and_project_output(context, w_o, b_o):
    merged = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(merged, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(Q, K, V, W_q, W_k, W_v, W_o, num_heads, attention_mask=None):
    B, T_q, D = Q.shape; T_k = K.shape[1]; d_h = D // num_heads; q = (Q @ W_q).reshape(B, T_q, num_heads, d_h).transpose(1, 2); k = (K @ W_k).reshape(B, T_k, num_heads, d_h).transpose(1, 2); v = (V @ W_v).reshape(B, T_k, num_heads, d_h).transpose(1, 2); scores = q @ k.transpose(-2, -1) / (d_h ** 0.5); scores = scores.masked_fill(attention_mask == 0, float('-inf')) if attention_mask is not None else scores; return ((torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T_q, D) @ W_o)

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, w1, b1), w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
def compute_layer_norm_mean_and_variance(x):
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, variance = compute_layer_norm_mean_and_variance(x)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return gamma * normalized + beta

# Step 37 - apply_residual_add_and_norm
def apply_residual_add_and_norm(x, sublayer_output, gamma, beta, eps=1e-5):
    return normalize_and_scale_with_gamma_beta(x + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return x * keep_mask.to(dtype=x.dtype) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, W_q, W_k, W_v, W_o, gamma, beta, num_heads, attention_mask=None):
    attention_output = assemble_multi_head_attention_forward(x, x, x, W_q, W_k, W_v, W_o, num_heads, attention_mask)
    return apply_residual_add_and_norm(x, attention_output, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    ffn_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, ffn_output, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask=None):
    x = encoder_layer_self_attention_sublayer(x, layer_params['w_q'], layer_params['w_k'], layer_params['w_v'], layer_params['w_o'], layer_params['attn_gamma'], layer_params['attn_beta'], num_heads, src_mask)
    return encoder_layer_feed_forward_sublayer(x, layer_params['w1'], layer_params['b1'], layer_params['w2'], layer_params['b2'], layer_params['ffn_gamma'], layer_params['ffn_beta'])

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask=None):
    return x if len(encoder_layer_params_list) == 0 else stack_encoder_layers(assemble_encoder_layer(x, encoder_layer_params_list[0], num_heads, src_mask), encoder_layer_params_list[1:], num_heads, src_mask)

# Step 43 - decoder_layer_masked_self_attention_sublayer
def decoder_layer_masked_self_attention_sublayer(y, W_q, W_k, W_v, W_o, gamma, beta, num_heads, tgt_mask=None):
    attention_output = assemble_multi_head_attention_forward(y, y, y, W_q, W_k, W_v, W_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, attention_output, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
def decoder_layer_cross_attention_sublayer(y, encoder_output, W_q, W_k, W_v, W_o, gamma, beta, num_heads, src_mask=None):
    attention_mask = src_mask[:, None, None, :] if src_mask is not None and src_mask.dim() == 2 else src_mask
    attention_output = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, W_q, W_k, W_v, W_o, num_heads, attention_mask)
    return apply_residual_add_and_norm(y, attention_output, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    ffn_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, ffn_output, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask=None, tgt_mask=None):
    y = decoder_layer_masked_self_attention_sublayer(y, layer_params['w_q_self'], layer_params['w_k_self'], layer_params['w_v_self'], layer_params['w_o_self'], layer_params['self_gamma'], layer_params['self_beta'], num_heads, tgt_mask)
    y = decoder_layer_cross_attention_sublayer(y, encoder_output, layer_params['w_q_cross'], layer_params['w_k_cross'], layer_params['w_v_cross'], layer_params['w_o_cross'], layer_params['cross_gamma'], layer_params['cross_beta'], num_heads, src_mask)
    return decoder_layer_feed_forward_sublayer(y, layer_params['w1'], layer_params['b1'], layer_params['w2'], layer_params['b2'], layer_params['ffn_gamma'], layer_params['ffn_beta'])

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask=None, tgt_mask=None):
    return y if len(decoder_layer_params_list) == 0 else stack_decoder_layers(assemble_decoder_layer(y, encoder_output, decoder_layer_params_list[0], num_heads, src_mask, tgt_mask), encoder_output, decoder_layer_params_list[1:], num_heads, src_mask, tgt_mask)

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_weight, output_bias=None):
    return apply_linear_projection(decoder_output, output_weight, output_bias)

# Step 49 - tie_output_projection_to_token_embeddings
def tie_output_projection_to_token_embeddings(token_embedding_weight):
    return token_embedding_weight.transpose(0, 1)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    return torch.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_token_ids, tgt_token_ids, model_params, num_heads, pad_id):
    embedding_weight = model_params['token_embedding']
    d_model = embedding_weight.shape[1]
    src_embeddings = scale_embeddings_by_sqrt_d_model(embedding_weight[src_token_ids], d_model)
    tgt_embeddings = scale_embeddings_by_sqrt_d_model(embedding_weight[tgt_token_ids], d_model)
    src_positional = build_sinusoidal_positional_encoding(src_token_ids.shape[1], d_model).to(device=src_embeddings.device, dtype=src_embeddings.dtype)
    tgt_positional = build_sinusoidal_positional_encoding(tgt_token_ids.shape[1], d_model).to(device=tgt_embeddings.device, dtype=tgt_embeddings.dtype)
    src_hidden = add_positional_encoding_to_embeddings(src_embeddings, src_positional)
    tgt_hidden = add_positional_encoding_to_embeddings(tgt_embeddings, tgt_positional)
    src_mask = build_padding_mask(src_token_ids, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_token_ids, pad_id)
    tgt_causal_mask = build_causal_mask(tgt_token_ids.shape[1]).to(tgt_token_ids.device)
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, tgt_causal_mask)
    encoder_output = stack_encoder_layers(src_hidden, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(tgt_hidden, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask)
    logits = apply_final_output_projection(decoder_output, model_params['output_projection'])
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    params = {}
    params['w_q'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w1'] = (torch.randn(d_model, d_ff) / math.sqrt(d_model)).requires_grad_()
    params['b1'] = torch.zeros(d_ff, requires_grad=True)
    params['w2'] = (torch.randn(d_ff, d_model) / math.sqrt(d_ff)).requires_grad_()
    params['b2'] = torch.zeros(d_model, requires_grad=True)
    params['attn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['attn_beta'] = torch.zeros(d_model, requires_grad=True)
    params['ffn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model, requires_grad=True)
    return params

# Step 53 - init_decoder_layer_parameters
def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    params = {}
    params['w_q_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_q_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w1'] = (torch.randn(d_model, d_ff) / math.sqrt(d_model)).requires_grad_()
    params['b1'] = torch.zeros(d_ff, requires_grad=True)
    params['w2'] = (torch.randn(d_ff, d_model) / math.sqrt(d_ff)).requires_grad_()
    params['b2'] = torch.zeros(d_model, requires_grad=True)
    params['self_gamma'] = torch.ones(d_model, requires_grad=True)
    params['self_beta'] = torch.zeros(d_model, requires_grad=True)
    params['cross_gamma'] = torch.ones(d_model, requires_grad=True)
    params['cross_beta'] = torch.zeros(d_model, requires_grad=True)
    params['ffn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model, requires_grad=True)
def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    params = {}
    params['w_q_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_q_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w1'] = (torch.randn(d_model, d_ff) / math.sqrt(d_model)).requires_grad_()
    params['b1'] = torch.zeros(d_ff, requires_grad=True)
    params['w2'] = (torch.randn(d_ff, d_model) / math.sqrt(d_ff)).requires_grad_()
    params['b2'] = torch.zeros(d_model, requires_grad=True)
    params['self_gamma'] = torch.ones(d_model, requires_grad=True)
    params['self_beta'] = torch.zeros(d_model, requires_grad=True)
    params['cross_gamma'] = torch.ones(d_model, requires_grad=True)
    params['cross_beta'] = torch.zeros(d_model, requires_grad=True)
    params['ffn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model, requires_grad=True)
def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    params = {}
    params['w_q_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o_self'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_q_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_k_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_v_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w_o_cross'] = (torch.randn(d_model, d_model) / math.sqrt(d_model)).requires_grad_()
    params['w1'] = (torch.randn(d_model, d_ff) / math.sqrt(d_model)).requires_grad_()
    params['b1'] = torch.zeros(d_ff, requires_grad=True)
    params['w2'] = (torch.randn(d_ff, d_model) / math.sqrt(d_ff)).requires_grad_()
    params['b2'] = torch.zeros(d_model, requires_grad=True)
    params['self_gamma'] = torch.ones(d_model, requires_grad=True)
    params['self_beta'] = torch.zeros(d_model, requires_grad=True)
    params['cross_gamma'] = torch.ones(d_model, requires_grad=True)
    params['cross_beta'] = torch.zeros(d_model, requires_grad=True)
    params['ffn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model, requires_grad=True)
    return params

# Step 54 - init_embedding_and_projection_parameters
def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    src_embedding = (torch.randn(vocab_size, d_model) / math.sqrt(d_model)).requires_grad_()
    tgt_embedding = (torch.randn(vocab_size, d_model) / math.sqrt(d_model)).requires_grad_()
    output_projection = tgt_embedding if tie_weights else (torch.randn(vocab_size, d_model) / math.sqrt(d_model)).requires_grad_()
    return {'src_embedding': src_embedding, 'tgt_embedding': tgt_embedding, 'output_projection': output_projection}

# Step 55 - collect_model_parameters_into_list
def collect_model_parameters_into_list(encoder_layer_params_list, decoder_layer_params_list, embedding_params):
    parameters = [parameter for layer in encoder_layer_params_list for parameter in layer.values()]
    parameters += [parameter for layer in decoder_layer_params_list for parameter in layer.values()]
    parameters += list(embedding_params.values())
    return list({id(parameter): parameter for parameter in parameters}.values())

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    start_tokens = torch.full((target_ids.shape[0], 1), start_token_id, dtype=target_ids.dtype, device=target_ids.device)
    return torch.cat((start_tokens, target_ids[:, :-1]), dim=1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    return float(d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5))

# Step 58 - build_uniform_smoothing_distribution
def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    return torch.full(shape, epsilon / (vocab_size - 2), dtype=torch.float32)

# Step 59 - set_confidence_on_gold_tokens
def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    result = smoothed_distribution.clone()
    return result.scatter_(-1, gold_token_ids.unsqueeze(-1), confidence)

# Step 60 - zero_pad_column_and_pad_token_rows
def zero_pad_column_and_pad_token_rows(distribution, gold_token_ids, pad_id):
    result = distribution.clone()
    result[..., pad_id] = 0
    result[gold_token_ids == pad_id] = 0
    return result

# Step 61 - compute_label_smoothed_kl_loss
def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    return (smoothed_distribution * -log_probabilities).sum()

# Step 62 - average_loss_over_non_pad_tokens
def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    non_pad_count = (gold_token_ids != pad_id).sum()
    return total_loss / non_pad_count.clamp_min(1)

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

