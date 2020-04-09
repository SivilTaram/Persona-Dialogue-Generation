import torch
import math
from agents.common.dict_helper import SpecialToken
import torch.nn.functional as F
from agents.transmitter.transmitter import Gpt2SeqModel
import numpy as np
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel


def _length_penalty(sequence_lengths):
    """https://arxiv.org/abs/1609.08144"""
    # return (5 + sequence_lengths) ** 0.9 / (5 + 1) ** 0.9
    # return torch.sqrt(sequence_lengths)
    return sequence_lengths


class LanguageModel(object):
    def __init__(self, pad_idx):
        self.transformer_module = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.transformer_module.eval()
        self.pad_idx = pad_idx

    def score_sentence(self, generate_tokens):
        batch_size, turn_size, sen_size = generate_tokens.size()
        generate_tokens = generate_tokens.view(batch_size * turn_size, sen_size)
        golden_out = generate_tokens[:, 1:].unsqueeze(dim=2)
        # TODO: manually construct the position ids for input & output
        with torch.no_grad():
            lm_logits, all_states = self.transformer_module(generate_tokens)
            # lm labels should mask the source sentence language model
            shift_logits = lm_logits[..., :-1, :].contiguous()
            # lm_labels = tgt_seq.clone()[..., 1:].contiguous()
            # predict answers
            scores = F.log_softmax(shift_logits, dim=2).gather(2, golden_out)
            nonzero = golden_out.ne(self.pad_idx).float()
            scores = (scores * nonzero).sum(1)
            seq_lengths = nonzero.sum(1)
            # Feature: calculate length penalty
            scores /= _length_penalty(seq_lengths)
            scores = scores.sum(dim=1)
        scores = scores.view(batch_size, turn_size)
        return scores

    def cuda(self, device):
        self.transformer_module.cuda(device)


def prepare_for_understand(send_messages, receive_messages, is_first_speaker, end_sep='__END__'):
    """Prepare input text for persona extraction
    :param send_messages, list of string, representing messages sended by the speaker
                          the index of list corresponds to the turn of the dialogue
    :param receive_messages, similar as send_messages
    :param is_first_speaker, bool, indicating if the speaker speaks first
    :param end_sep, string, marking the end of the received text
    """
    batch_size = len(send_messages[0])

    text = [""] * len(send_messages[0])
    valid_mask = []
    for batch_send, batch_rec in zip(send_messages, receive_messages):
        # text += s + ' ' + start_sep + ' ' + r + ' ' + end_sep + ' '
        for ind, send in enumerate(batch_send):
            text[ind] += ' ' + send + ' ' + end_sep + ' '
        valid_mask.append([1] * batch_size)
        for ind, send in enumerate(batch_rec):
            text[ind] += ' ' + send + ' ' + end_sep + ' '
        valid_mask.append([0] * batch_size)

    return text, valid_mask