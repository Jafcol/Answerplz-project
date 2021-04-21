from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from psutil import cpu_count
from os import environ
# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = 'ACTIVE'
from onnxruntime_tools.transformers.gpt2_helper import Gpt2Helper
from onnxruntime.capi._pybind_state import set_seed as ort_set_seed
import onnxruntime as ort
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy
from typing import Iterable, List, Optional, Tuple, Any
import random

@torch.no_grad()
def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> List:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


@torch.no_grad()
def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


@torch.no_grad()
def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []
    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False
    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []
        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )
            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue
            banned_tokens_slice.append(banned_token_seq[-1])
        banned_tokens.append(banned_tokens_slice)
    return banned_tokens


@torch.no_grad()
def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]].
    banned_mask = torch.sparse.Tensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class GPT2ONNXModel:
    def __init__(self, onnx_model_path, gpt2_model_path, device='cuda', verbose=False, threads=None, is_float16=False):
        self.config = AutoConfig.from_pretrained(gpt2_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_attention_heads = self.config.n_head
        self.hidden_size = self.config.n_embd
        self.num_layer = self.config.n_layer
        self.verbose = verbose
        self.is_float16 = is_float16
        # Set the seed for onnxruntime.
        torch.random.manual_seed(random.randint(1, 9999999))
        torch.cuda.manual_seed_all(random.randint(1, 9999999))
        torch.manual_seed(random.randint(1, 9999999))
        numpy.random.seed(random.randint(1, 9999999))
        ort.set_seed(random.randint(1, 9999999))
        # Set our session options.
        options = ort.SessionOptions()
        # set session threads if user supplied.
        if threads is not None:
            options.inter_op_num_threads = threads
            options.intra_op_num_threads = threads
        # options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # Only CPU supports parallel.
        if device == 'cpu':
            options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        # Start the ONNX session.
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=options,
            providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider']
        )
        self.device = device
        if self.verbose is True:
            print('ONNX Session Created!')

    @torch.no_grad()
    def generate(self, input_text, max_length: Optional[int] = None, min_length: Optional[int] = None, do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None, num_beams: Optional[int] = None, temperature: Optional[float] = None,
        top_k: Optional[int] = None, top_p: Optional[float] = None, repetition_penalty: Optional[float] = None, bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None, length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None, num_return_sequences: Optional[int] = None, use_cache: Optional[bool] = None,
        **model_kwargs
    ) -> torch.LongTensor:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.
        """

        if len(input_text) > 1024:
            input_text = input_text[-1024:]

        # Get inputs from input text. (input ids, attention mask, position, and past)
        input_ids, attention_mask, position_ids, past = self.get_inputs(input_text)
        # Get user supplied text generation options. Replace with model config on unset options.
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1
        # Make sure all options were set correctly.
        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        max_length = len(input_ids[0]) + max_length

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        # current position and vocab size
        vocab_size = self.config.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure " \
           f"that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` " \
           f"or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids, cur_len=cur_len, max_length=max_length, min_length=min_length, do_sample=do_sample,
                early_stopping=early_stopping, temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id,
                batch_size=effective_batch_size, num_return_sequences=num_return_sequences,
                length_penalty=length_penalty, num_beams=num_beams, vocab_size=vocab_size,
                attention_mask=attention_mask, past=past, position_ids=position_ids, use_cache=use_cache,
                model_kwargs=model_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids, cur_len=cur_len, max_length=max_length, min_length=min_length, do_sample=do_sample,
                temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, pad_token_id=pad_token_id,
                eos_token_id=eos_token_id, batch_size=effective_batch_size, attention_mask=attention_mask,
                use_cache=use_cache, model_kwargs=model_kwargs,
            )

        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output

    @torch.no_grad()
    def _generate_no_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, temperature,
        top_k, top_p, repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id,
        batch_size, attention_mask, use_cache, model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past_shape = [2, input_ids.shape[0], self.num_attention_heads, 0, self.hidden_size // self.num_attention_heads]
        past = []
        for i in range(self.num_layer):
            past.append(torch.empty(past_shape, dtype=torch.float32).to(self.device))
        attention_mask = attention_mask.clone().detach()
        
        while cur_len < max_length:

            position_ids = (attention_mask.long().cumsum(-1) - 1)
            position_ids.masked_fill_(position_ids < 0, 0)

            outputs = self.inference_with_io_binding(
                input_ids.to(self.device),
                position_ids.to(self.device),
                attention_mask.to(self.device),
                past
            )
            next_token_logits = outputs[0][:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits, input_ids=input_ids, no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids, cur_len=cur_len, min_length=min_length, max_length=max_length,
                eos_token_id=eos_token_id, repetition_penalty=repetition_penalty, batch_size=batch_size, num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            # if "past_key_values" in outputs:
            #     past = outputs.past_key_values
            # elif "mems" in outputs:
            #     past = outputs.mems

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = self.top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return input_ids

    @torch.no_grad()
    def _generate_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, early_stopping, temperature,
        top_k, top_p, repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id, batch_size,
        num_return_sequences, length_penalty, num_beams, vocab_size, attention_mask, past, position_ids, use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example with beam search."""

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # done sentences
        done = [False for _ in range(batch_size)]

        past_shape = [2, batch_size * num_beams, self.num_attention_heads, 0, self.hidden_size // self.num_attention_heads]
        past = []
        for i in range(self.num_layer):
            past.append(torch.empty(past_shape, dtype=torch.float32).to(self.device))

        next_tokens = []
        while cur_len < max_length:
            position_ids = (attention_mask.long().cumsum(-1) - 1)
            position_ids.masked_fill_(position_ids < 0, 0)
            # outputs = self(**model_inputs, return_dict=True)  # (batch_size * num_beams, cur_len, vocab_size)
            outputs = self.inference_with_io_binding(
                input_ids.to(self.device),
                position_ids.to(self.device),
                attention_mask.to(self.device),
                past
            )
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            # if "past_key_values" in outputs:
            #     past = outputs.past_key_values
            # elif "mems" in outputs:
            #     past = outputs.mems

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            scores = self.postprocess_next_token_scores(scores=scores, input_ids=input_ids, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids,
                cur_len=cur_len, min_length=min_length, max_length=max_length, eos_token_id=eos_token_id, repetition_penalty=repetition_penalty,
                batch_size=batch_size, num_beams=num_beams,
            )

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = self.top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(),
                            beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx],
                    beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(self.device)

        return decoded

    @torch.no_grad()
    def get_inputs(self, prompt_text):
        encodings_dict = self.tokenizer.batch_encode_plus([prompt_text], padding=True)
        input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64, device=self.device)
        attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32, device=self.device)
        position_ids = (attention_mask.long().cumsum(-1) - 1).to(self.device)
        position_ids.masked_fill_(position_ids < 0, 0)
        # Empty Past State for generating first word
        batch_size = input_ids.size(0)
        past_shape = [2, batch_size, self.num_attention_heads, 0, self.hidden_size // self.num_attention_heads]
        empty_past = []
        for i in range(self.num_layer):
            empty_past.append(torch.empty(past_shape, dtype=torch.float32).to(self.device))
        return input_ids, attention_mask, position_ids, empty_past

    @torch.no_grad()
    def inference_with_io_binding(self, input_ids, position_ids, attention_mask, past):
        """
        For performing gpt2 inference using our persistant session.
        """
        output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=input_ids.size(0), past_sequence_length=past[0].size(3),
            sequence_length=input_ids.size(1), config=self.config
        )
        output_buffers = Gpt2Helper.get_output_buffers(output_shapes, self.device, is_float16=self.is_float16)
        io_binding = Gpt2Helper.prepare_io_binding(
            self.session, input_ids, position_ids, attention_mask,
            past, output_buffers, output_shapes
        )
        self.session.run_with_iobinding(io_binding)
        outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(
            self.session, output_buffers, output_shapes,
            return_numpy=False
        )
        return outputs

    @torch.no_grad()
    def top_k_top_p_filtering(self, logits: Tensor, top_k: int = 0, top_p: float = 1.0,
        filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1,
    ) -> torch.Tensor:
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    @torch.no_grad()
    def postprocess_next_token_scores(self, scores, input_ids, no_repeat_ngram_size, bad_words_ids, cur_len, min_length,
        max_length, eos_token_id, repetition_penalty, batch_size, num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # Exclude EOS token (already processed)
            bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
            # Modify the scores in place by setting the banned tokens logits to `-inf`
            set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

        return scores

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Any]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)