from parlai.core.dict import DictionaryAgent
import copy
from parlai.core.params import ParlaiParser
from parlai.scripts.build_dict import build_dict
from collections import defaultdict
import os


class SpecialToken:
    start = '<start>'
    end = '<end>'
    # compatible with gpt2 vocabulary
    pad = '<unk>'
    unk = '<unk>'
    # transmitter
    persona_start = '<pstart>'
    persona_end = '<pend>'
    talk_1_start = '<t1start>'
    talk_1_end = '<t1end>'
    talk_2_start = '<t2start>'
    talk_2_end = '<t2end>'
    no_fact = '<nofact>'
    # smoother
    class_sym = '<cls>'
    sep_sym = '<sep>'
    slice_sym = '__slice__'


def build_transmitter_dict():
    def setup_args():
        parser = ParlaiParser(add_model_args=True, add_parlai_args=True)
        parser.set_defaults(task='tasks.convai2transmitter.agents:BothTeacher',
                            dict_initpath='../../tmp/dict/init_transmitter.dict',
                            datatype='train',
                            dict_lower=True,
                            dict_file='../../tmp/dict/convai2_self_seq2seq_model.dict',
                            dict_nulltoken=SpecialToken.pad,
                            dict_starttoken=SpecialToken.start,
                            dict_endtoken=SpecialToken.end,
                            dict_unktoken=SpecialToken.unk,
                            dict_tokenizer='split',
                            dict_language='english',
                            dict_include_valid=True,
                            dict_minfreq=2,
                            dict_maxexs=-1)
        return parser
    parser = setup_args()
    opt = parser.parse_args(args=[])
    # %%
    build_dict(opt)


def build_receiver_dict():
    def setup_args():
        parser = ParlaiParser(add_model_args=True, add_parlai_args=True)
        parser.set_defaults(task='tasks.convai2receiver.agents:BothOriginalTeacher',
                            dict_initpath='../../tmp/dict/init_receiver.dict',
                            datatype='train',
                            dict_lower=True,
                            dict_file='../../tmp/dict/receiver_origin.dict',
                            dict_tokenizer='split',
                            dict_language='english',
                            dict_include_valid=True,
                            dict_maxexs=-1)
        return parser
    parser = setup_args()
    opt = parser.parse_args(args=[])
    # %%
    build_dict(opt)


def build_smoother_dict():
    def setup_args():
        parser = ParlaiParser(add_model_args=True, add_parlai_args=True)
        parser.set_defaults(task='tasks.convai2smoother.agents:BothOriginalTeacher',
                            # dict_initpath='../../tmp/dict/init_receiver.dict',
                            datatype='train',
                            dict_lower=True,
                            dict_file='../../tmp/dict/smoother_origin.dict',
                            dict_tokenizer='split',
                            dict_language='english',
                            dict_include_valid=True,
                            dict_maxexs=-1)
        return parser
    parser = setup_args()
    opt = parser.parse_args(args=[])
    # %%
    build_dict(opt)


if __name__ == '_main_':
    build_transmitter_dict()
    # build_receiver_dict()
    # build_smoother_dict()