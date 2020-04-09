#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.dict import DictionaryAgent
import torch
try:
    from pytorch_pretrained_bert import OpenAIGPTTokenizer
except ImportError:
    raise ImportError('please ensure that pytorch-pretrained-BERT installed. \n '
                      'pip install pytorch-pretrained-bert')
from .dict_helper import SpecialToken
import os


def recover_bpe_encoding(bpe_tokens):
    output_tokens = []
    temp_token = ''
    for bpe_token in bpe_tokens:
        if '</w>' in bpe_token:
            temp_token += bpe_token.replace('</w>', '')
            output_tokens.append(temp_token)
            temp_token = ''
        else:
            temp_token += bpe_token.strip()
    if temp_token != '':
        output_tokens.append(temp_token)
    return output_tokens


class GPTDictionaryAgent(DictionaryAgent):
    """ Allow to use the Torch Agent with the wordpiece dictionary of Hugging Face.
    """
    default_lang = 'english'
    default_maxngram = -1
    default_minfreq = 0
    default_maxtokens = -1
    default_null = SpecialToken.pad
    default_start = SpecialToken.start
    default_end = SpecialToken.end
    default_unk = SpecialToken.unk
    default_tok = 'bpe'
    default_lower = True
    default_textfields = 'text,labels'

    def __init__(self, opt):
        super().__init__(opt)
        # initialize from voab path
        cache_vocab_dir = os.path.join(opt['datapath'], 'models', 'gpt_models')
        self.special_tokens = [SpecialToken.talk_1_start,
                               SpecialToken.talk_1_end,
                               SpecialToken.persona_start,
                               SpecialToken.persona_end,
                               SpecialToken.no_fact,
                               SpecialToken.start,
                               SpecialToken.end,
                               SpecialToken.slice_sym]

        # add special token after the pre-trained bpe text
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt',
                                                            cache_dir=cache_vocab_dir,
                                                            special_tokens=self.special_tokens)

        self.start_token = self.default_start
        self.end_token = self.default_end
        self.null_token = self.default_null

        # <unk> already in the dictionary
        self.start_idx = self.tokenizer.convert_tokens_to_ids([SpecialToken.start])[0]
        # <end> is used to split a long text into different parts, which is necessary for us
        # to differentiate persona & history only passing the observation function for one time
        self.end_idx = self.tokenizer.convert_tokens_to_ids([SpecialToken.end])[0]
        self.pad_idx = self.tokenizer.convert_tokens_to_ids([SpecialToken.pad])[0]  # should be 0
        # update for default tokenizer vocabulary
        self.tok2ind.clear()
        self.ind2tok.clear()

        # set tok2ind for special tokens
        for special_token in self.special_tokens + [self.start_token, self.end_token, self.null_token]:
            token_id = self.tokenizer.convert_tokens_to_ids([special_token])[0]
            self.tok2ind[special_token] = token_id
            self.ind2tok[token_id] = special_token

    def txt2vec(self, text, vec_type=list):
        tokens = self.tokenizer.tokenize(text)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id

    def vec2txt(self, tensor_list, recover_bpe=False):
        if isinstance(tensor_list[0], torch.Tensor):
            idxs = [idx.cpu().item() for idx in tensor_list]
        else:
            idxs = list(tensor_list)
        # filter unk ids
        max_vocab_size = len(self.tokenizer.decoder) + len(self.special_tokens)
        idxs = [self.pad_idx if idx >= max_vocab_size else idx for idx in idxs]
        toks = self.tokenizer.convert_ids_to_tokens(idxs)
        if recover_bpe:
            toks = recover_bpe_encoding(toks)
        return ' '.join(toks)
