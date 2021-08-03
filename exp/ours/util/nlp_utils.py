import torch
from typing import Dict, List

from transformers import PreTrainedTokenizer, BatchEncoding, T5Tokenizer, BartTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.modeling_outputs import Seq2SeqLMOutput

from exp.ours.util import py_utils
from torch.nn import functional as F


def tokenize_with_cache(text, tokenizer, cache):
  out = []
  for word in text.split():
    cached = cache.get(word)
    if cached is None:
      cached = tokenizer.tokenize(word)
      cache[word] = cached
    out += cached
  return out


def encode_with_cache(text, tokenizer, cache) -> List[int]:
  out = []
  for word in text.split():
    cached = cache.get(word)
    if cached is None:
      cached = tokenizer.encode(word, add_special_tokens=False)
      cache[word] = cached
    out += cached
  return out


def prepare_batch_from_pre_encoded(
    encoded_text: List[List[int]], tokenizer: PreTrainedTokenizer, max_length, truncation=False):
  """Effectively runs `tokenizer.__call__` on text that has been pre-tokenized

  We use this to support pre-tokenizing the input text before collating
  """

  batch_outputs = {}
  for text_ids in encoded_text:
    outputs = tokenizer.prepare_for_model(
      text_ids,
      None,
      add_special_tokens=False,
      padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
      truncation=truncation,
      max_length=max_length,
    )

    for key, value in outputs.items():
      if key not in batch_outputs:
        batch_outputs[key] = []
      batch_outputs[key].append(value)

  batch_outputs = tokenizer.pad(
    batch_outputs,
    padding=PaddingStrategy.LONGEST.value,
    max_length=max_length,
  )
  return BatchEncoding(batch_outputs, tensor_type="pt").data


def t5_initialize_decoding(tokenizer, model, encoder_out, encoder_mask, post_process=None):
  batch_size = encoder_out.size(0)
  device = encoder_out.device
  initial_state = dict(
    encoder_mask=encoder_mask,
    encoder_outputs=encoder_out
  )

  def _decode_step(predictions, prev_state, time_step):
    return _t5_decoding_step(model, predictions, prev_state, post_process, time_step)

  initial_out = torch.full(
    (batch_size,), tokenizer.pad_token_id, dtype=torch.long, device=device)

  return initial_out, initial_state, _decode_step


def _t5_decoding_step(model, predictions, state, post_process, time_step):
  past = py_utils.flat_to_nested_struct({k: v.contiguous() for k, v in state.items()
                                         if isinstance(k, tuple)})
  model_inputs = model.prepare_inputs_for_generation(
    predictions.unsqueeze(1),
    past=past, attention_mask=state["encoder_mask"],
    encoder_outputs=(state["encoder_outputs"],),
    use_cache=True)
  out = model(**model_inputs, return_dict=True)
  logits = out.logits

  logits = logits.squeeze(1)
  logits = F.log_softmax(logits, -1)

  if post_process is not None:
    logits = post_process(logits, model_inputs, time_step)

  next_state = dict(
    encoder_mask=state["encoder_mask"],
    encoder_outputs=state["encoder_outputs"],
  )
  py_utils.nested_struct_to_flat(out.past_key_values, cur_dict=next_state)
  return logits, next_state


def get_tokens_with_space(tokenizer):
  if isinstance(tokenizer, T5Tokenizer):
    global _t5_tokens_with_spaces
    if _t5_tokens_with_spaces is None:
      start_of_word_tokens_ix = []
      for k, ix in tokenizer.get_vocab().items():
        if k.startswith("▁"):
          start_of_word_tokens_ix.append(ix)
      start_of_word_tokens_ix = torch.tensor(start_of_word_tokens_ix, dtype=torch.long)
      _t5_tokens_with_spaces = start_of_word_tokens_ix
    return _t5_tokens_with_spaces

  elif isinstance(tokenizer, BartTokenizer):
    global _bart_tokens_with_spaces
    if _bart_tokens_with_spaces is None:
      start_of_word_tokens_ix = []
      for k, ix in tokenizer.get_vocab().items():
        if k.startswith("Ġ"):
          start_of_word_tokens_ix.append(ix)
      start_of_word_tokens_ix = torch.tensor(start_of_word_tokens_ix, dtype=torch.long)
      _bart_tokens_with_spaces = start_of_word_tokens_ix
    return _bart_tokens_with_spaces

  else:
    raise NotImplementedError()


def get_starts_word(tokenizer, voc_size=None):
  if voc_size is None:
    voc_size = len(tokenizer)
  start_of_word_tokens_ix = get_tokens_with_space(tokenizer)
  starts_new_word = torch.zeros(voc_size, dtype=torch.bool)
  starts_new_word[start_of_word_tokens_ix] = 1
  return starts_new_word


def t5_initialize_decoding_boost(
    tokenizer, model, encoder_out, encoder_mask,
    boost_targets,
):
  batch_size = encoder_out.size(0)
  device = encoder_out.device
  initial_state = dict(
    encoder_mask=encoder_mask,
    encoder_outputs=encoder_out,
    batch_id=torch.arange(0, batch_size, device=device, dtype=torch.long),
    num_word_pieces=torch.zeros(batch_size, device=device, dtype=torch.long),
    on_banned=torch.zeros(batch_size, len(boost_targets), device=device, dtype=torch.bool)
  )

  starts_new_word = get_starts_word(tokenizer, model.config.vocab_size).to(device)

  boost_targets = [tokenizer.encode_with_cache(x, add_special_tokens=False) for x in boost_targets]
  boost_tensor = torch.full(
    (batch_size, max(len(x) for x in boost_targets)+1),
    fill_value=-1,
    dtype=torch.long
  )
  for i, target in enumerate(boost_targets):
    boost_tensor[i, :len(target)] = target
  boost_tensor = boost_tensor.to(device)
  boost_lengths = torch.tensor(
    [len(x) for x in boost_targets],
    device=device, dtype=torch.long
  )

  def _decode_step(predictions, prev_state):
    return _t5_boost_decoding_step(model, predictions, prev_state,
                             starts_new_word, boost_tensor, boost_lengths)

  initial_out = torch.full(
    (batch_size,), tokenizer.pad_token_id, dtype=torch.long, device=device)

  return initial_out, initial_state, _decode_step


def _t5_boost_decoding_step(model, predictions, state, starts_new_word,
                            boost_tensor, boost_lengths):
  batch_ids = state["batch_id"]

  # Compute what word piece number we just generated, indexed at 0
  generated_space = starts_new_word[predictions]
  on_piece = state["num_word_pieces"]
  on_piece = torch.logical_not(generated_space).to(torch.long)*(1 + on_piece)
  n_peices, n_word = boost_tensor.size()
  on_piece = torch.min(on_piece, torch.as_tensor(n_peices-1).to(on_piece))

  past = py_utils.flat_to_nested_struct({
    k: v.contiguous() for k, v in state.items() if isinstance(k, tuple)})
  model_inputs = model.prepare_inputs_for_generation(
    predictions.unsqueeze(1), past=past, decoder_attention_mask=state["encoder_mask"],
    encoder_outputs=(state["encoder_outputs"],),
    use_cache=True)
  out = model(**model_inputs, return_dict=True)
  logits = out.logits

  logits = logits.squeeze(1)
  logits = F.log_softmax(logits, -1)

  # on_boost[x, y] is true if beam x currently match boosted word y
  on_boost = state["on_banned"]   # [batch, n_words]
  on_boost[generated_space] = True  # Start off matching all boosted words
  # Now require that the last prediction matched
  next_boosted_word_piece = on_boost[state["batch_id"], on_piece]
  on_boost = torch.logical_and(next_boosted_word_piece == predictions.unsqueeze(1), on_boost)

  # For each beam, has the beam just completed a boosted word
  completed_boosted_word = torch.any(torch.logical_and(
    on_boost, boost_lengths == (on_piece+1).unsqueeze(1)), 1)

  # A beam that has completed a boosted words boosts any token that would start a new word
  do_boost = torch.logical_and(
    completed_boosted_word.unsqueeze(1),
    starts_new_word.unsqueeze(0)
  )  # [batch, voc_size]

  ixs = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
  next_word_piece = boost_tensor[ixs, on_piece+1]  # [batch, n_words]
  next_word_piece = torch.where(on_boost, next_word_piece, torch.full_like(next_word_piece, -1))


  next_state = dict(
    encoder_mask=state["encoder_mask"],
    encoder_outputs=state["encoder_outputs"],
    num_word_pieces=on_piece,
    num_spaces=on_word,
    batch_id=batch_ids
  )
  py_utils.nested_struct_to_flat(out.past_key_values, cur_dict=next_state)
  return logits, next_state