#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.dict import DictionaryAgent
from parlai.core.agents import Agent
from agents.common.dict_helper import SpecialToken
import random
import torch
from torch import optim
import torch.nn as nn

from collections import deque, defaultdict
import json
import os
import math
import pickle
from agents.receiver.receiver import split_pad_vector
# basic setting, we should make it easy to convert from basic setting into gpt/bert setting
from agents.transmitter.utils import maintain_dialog_history, PaddingUtils, round_sigfigs
from agents.transmitter.utils import SharedTable
from agents.transmitter.utils import modelzoo_path
from agents.transmitter.gpt.loss import LabelSmoothingLoss, TokenCrossEntropyLoss
from agents.transmitter.seq2seq.model import Seq2seqModel
from .gpt.model import Gpt2SeqModel
from .gpt.optim import GPTOptimizer
from agents.common.gpt_dictionary import GPTDictionaryAgent

# lstm, transformer, gpt2
ARCH_CHOICE = 'gpt'


def print_model(model):
    print(model)
    k = 0
    params = list(model.parameters())
    for i in params:
        if i.requires_grad:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
    print("Total Parameter Size：" + str(k))


def transformer_args(agent_args):
    agent_args.add_argument('--n_layers', type=int, default=2)
    agent_args.add_argument('--embedding_size', type=int, default=768)
    agent_args.add_argument('--inner_size', type=int, default=2048)
    agent_args.add_argument('--key_size', type=int, default=64)
    agent_args.add_argument('--value_size', type=int, default=64)
    agent_args.add_argument('--n_head', type=int, default=12)
    agent_args.add_argument('--dropout', type=float, default=0.1)
    agent_args.add_argument('--max_seq_len', type=int, default=256)


def lstm_args(agent_args):
    agent_args.add_argument('--lr', type=float, default=3.0, help='fine-tuning learning rate')
    agent_args.add_argument('--dropout', type=float, default=0.1)
    # rank candidates in validation
    agent_args.add_argument('--attention-time', default='post')
    agent_args.add_argument('--attention-length', default=-1)

    agent_args.add_argument('--rnn-class', default='lstm')

    # none, shared
    agent_args.add_argument('--rnn-share', default='none')
    agent_args.add_argument('--beam-size', default=1)

    agent_args.add_argument('--encoder-embed-dim', type=int, default=512)

    agent_args.add_argument('--encoder-hidden-size', type=int, default=1024)
    agent_args.add_argument('--encoder-layers', type=int, default=2)
    agent_args.add_argument('--encoder-bidirectional', type=bool, default=False)

    agent_args.add_argument('--decoder-embed-dim', type=int, default=512)
    agent_args.add_argument('--decoder-hidden-size', type=int, default=1024)
    agent_args.add_argument('--decoder-layers', type=int, default=1)
    agent_args.add_argument('--decoder-attention', type=str, default='general')
    # Granular dropout settings (if not specified these default to --dropout)
    agent_args.add_argument('--encoder-dropout-in', type=float, default=0.1)
    agent_args.add_argument('--encoder-dropout-out', type=float, default=0.1)
    agent_args.add_argument('--decoder-dropout-in', type=float, default=0.1)
    agent_args.add_argument('--decoder-dropout-out', type=float, default=0.1)
    # decoder input and output share the same vocabulary
    agent_args.add_argument('--share-decoder-input-output-embed', type=bool, default=True)
    agent_args.add_argument('--gradient_clip', type=float, default=5.0)


def gpt2_args(agent_args):
    # do not any thing
    agent_args.add_argument('--lr', type=float, default=6.25e-5)
    agent_args.add_argument('--gradient_clip', type=int, default=1.0)
    agent_args.add_argument('--warmup_proportion', type=float, default=0.002)
    agent_args.add_argument('--lr_schedule', type=str, default='warmup_linear')
    agent_args.add_argument('--weight_decay', type=float, default=0.01)
    agent_args.add_argument('--lm_coef', type=float, default=0.9)
    agent_args.add_argument('--num_train_epochs', type=int, default=3)
    # default size for num_train_epochs
    agent_args.add_argument('--train_size', type=int, default=130824)
    agent_args.add_argument('--optimizer_step', type=int, default=5000)


class TransformerAgent(Agent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model currently uses greedy decoding, selecting the
    highest probability token at each time step.

    For more information, see the following papers:
    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
        'gpt_custom': GPTOptimizer
    }

    @staticmethod
    def dictionary_class():
        if ARCH_CHOICE == 'gpt':
            return GPTDictionaryAgent
        elif ARCH_CHOICE == 'lstm':
            return DictionaryAgent
        else:
            raise NotImplementedError("Do not support the dictionary agent for arch: {}".format(ARCH_CHOICE))

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent_args = argparser.add_argument_group('Seq2Seq Arguments')
        # use lstm encoder-decoder or transformer
        agent_args.add_argument('-gpu', '--gpu', type=int, default=-1,
                                help='which GPU to use')
        agent_args.add_argument('--no-cuda', type=bool, default=False,
                                help='disable GPUs even if available. otherwise, will use GPUs if '
                                     'available on the device.')
        agent_args.add_argument('--rank_candidates', type=bool, default=False,
                                help='Whether the model should parse candidates for ranking.')
        agent_args.add_argument('-emb', '--embedding-type', default='glove-fixed',
                                choices=['random', 'glove', 'glove-fixed'],
                                help='Choose between different strategies '
                                     'for word embeddings. Default is random, '
                                     'but can also preinitialize from Glove or '
                                     'Fasttext.'
                                     'Preinitialized embeddings can also be fixed '
                                     'so they are not updated during training.')
        agent_args.add_argument('-lt', '--lookuptable', default='all',
                                choices=['unique', 'enc_dec', 'dec_out', 'all'],
                                help='The encoder, decoder, and output modules can '
                                     'share weights, or not. '
                                     'Unique has independent embeddings for each. '
                                     'Enc_dec shares the embedding for the encoder '
                                     'and decoder. '
                                     'Dec_out shares decoder embedding and output '
                                     'weights. '
                                     'All shares all three weights.')

        agent_args.add_argument('-rf', '--report_freq', type=float, default=0.001)
        agent_args.add_argument('--smoothing', type=float, default=0.0, help='smoothing loss for transformer')

        """ Add option for training """
        agent_args.add_argument('--n_epoches', type=int, default=100, help='maximum training epochs')
        agent_args.add_argument('--batchsize', type=int, default=256,
                                help='batch size for training sequence to sequence')
        agent_args.add_argument('--optimizer', default='gpt_custom', help='optimizer, defined in OPTIM_OPTS')
        agent_args.add_argument('--momentum', default=0.9, help='momentum in sgd optimizer')
        agent_args.add_argument('--display_model', type=bool, default=True, help='whether display the model')

        agent_args.add_argument('--history-replies', default='label_else_model')

        agent_args.add_argument('--share-encoder-persona-dialogue', type=bool, default=True,
                                help='share the same encoder when encoding dialogue and persona')
        agent_args.add_argument('--encoder_dis_use', type=bool, default=False,
                                help='add distance state embedding')
        agent_args.add_argument('--encoder_turn_use', type=bool, default=False,
                                help='add turn state embedding')
        agent_args.add_argument('--encoder_turn_dim', type=int, default=0,
                                help='encoder turn dimension')

        agent_args.add_argument('--use-persona-token', type=bool, default=True,
                                help='add special tokens at the start and end of persona')
        agent_args.add_argument('--use-talk-token', type=bool, default=True,
                                help='add special tokens at the start and end of query')
        agent_args.add_argument('--persona-append-strategy', default='concat', choices=['concat', 'none', 'select'],
                                help='add special tokens at the start and end of query')

        agent_args.add_argument('--encode_max_seq_len', type=int, default=184)
        agent_args.add_argument('--decode_max_seq_len', type=int, default=32)

        # new setting on persona gate, which is implemented by Receiver
        agent_args.add_argument('--select-persona', default=False,
                                help='use receiver to select persona to genreate the response')
        agent_args.add_argument('--receiver-model', default='',
                                help='receiver model for selecting persona, else append default ones')
        agent_args.add_argument('--shuffle-persona', default=False,
                                help='shuffle the order of persona to make model robust ')

        agent_args.add_argument('--history-append-strategy', type=int, default=-1,
                                help='-1 means all history are appended, and 0 means none, either')

        agent_args.add_argument('--beam_size', type=int, default=3)

        """ Add option for model """
        if ARCH_CHOICE == 'lstm':
            lstm_args(agent_args)
        elif ARCH_CHOICE == 'transformer':
            transformer_args(agent_args)
        elif ARCH_CHOICE == 'gpt':
            gpt2_args(agent_args)
        else:
            print("Not support architecture : {} !".format(ARCH_CHOICE))
            exit(-1)

        """ Add option for dictionary agent"""
        agent_args.add_argument('--dict-nulltoken', default=SpecialToken.pad)
        agent_args.add_argument('--dict-starttoken', default=SpecialToken.start)
        agent_args.add_argument('--dict-endtoken', default=SpecialToken.end)
        agent_args.add_argument('--dict-unktoken', default=SpecialToken.unk)
        agent_args.add_argument('--dict-tokenizer', default='split')
        agent_args.add_argument('--dict-language', default='english')
        agent_args.add_argument('--dict-include-valid', type=bool, default=True)
        agent_args.add_argument('--dict_file', default='../../tmp/dict/convai2_self_seq2seq_model.dict')
        agent_args.add_argument('--dict_lower', type=bool, default=True)

        TransformerAgent.dictionary_class().add_cmdline_args(argparser)
        return agent_args

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        opt = self.opt  # there is a deepcopy in the init

        # all instances may need some params
        self.encode_max_seq_len = opt['encode_max_seq_len'] if opt['encode_max_seq_len'] > 0 else None
        self.decode_max_seq_len = opt['decode_max_seq_len'] if opt['decode_max_seq_len'] > 0 else None

        self.metrics = {'loss': 0.0, 'num_tokens': 0, 'correct_tokens': 0, 'total_skipped_batches': 0, 'correct_pred': 0, 'pred_count': 0}

        self.history = {}
        # batch share the same persona information
        self.use_person_tokens = opt.get('use_persona_token', False)
        self.use_talk_tokens = opt.get('use_talk_token', False)
        self.use_history_reply = opt.get('history_replies', 'label_else_model')
        self.add_default_persona = opt.get('add_default_persona', True)
        self.persona_append_strategy = opt.get('persona_append_strategy', 'concat')
        self.history_append_strategy = opt.get('history_append_strategy', -1)

        self.report_freq = opt.get('report_freq', 0.001)
        self.batch_idx = shared and shared.get('batchindex') or 0
        self.rank = opt['rank_candidates']
        self.beam_size = opt.get('beam_size', 1)
        self.topk = opt.get('topk', 1)
        states = {}

        # if gpt2
        if 'gpt' in ARCH_CHOICE:
            num_optim_steps = opt['train_size'] * opt['num_train_epochs'] // opt['batchsize']
            # override optimizer_step
            opt['optimizer_step'] = num_optim_steps

        # check for cuda
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()

        if shared:
            # set up shared properties
            self.opt = shared['opt']
            opt = self.opt
            self.dict = shared['dict']
            self.START_IDX = shared['START_IDX']
            self.END_IDX = shared['END_IDX']
            self.NULL_IDX = shared['NULL_IDX']
            # answers contains a batchsize list of the last answer produced
            self.answers = shared['answers']
            self.model = shared['model']
            self.metrics = shared['metrics']
            self.receiver = shared['receiver']
            self.receiver_dict = shared['receiver_dict']
            states = shared.get('states', {})
        else:
            # this is not a shared instance of this class, so do full init
            # answers contains a batchsize list of the last answer produced
            self.answers = [None] * opt['batchsize']

            if self.use_cuda:
                print('[ Using CUDA ]')
                torch.cuda.set_device(opt['gpu'])

            init_model = None
            # check first for 'init_model' for loading model from file
            if opt.get('init_model') and os.path.isfile(opt['init_model']):
                init_model = opt['init_model']
            # next check for 'model_file', this would override init_model
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                init_model = opt['model_file']

            if init_model is not None:
                # load model parameters if available
                print('[ Loading existing model params from {} ]'.format(init_model))
                states = self.load(init_model)

                if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                    opt['dict_file'] = init_model + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = self.dictionary_class()(opt)
            self.id = 'Transformer'
            # we use START markers to start our output
            self.START_IDX = self.dict[self.dict.start_token]
            # we use END markers to end our output
            self.END_IDX = self.dict[self.dict.end_token]
            # get index of null token from dictionary (probably 0)
            self.NULL_IDX = self.dict[self.dict.null_token]

            # get vocab size
            vocab_size = len(self.dict.tok2ind.items())

            if ARCH_CHOICE == 'lstm':
                self.model = Seq2seqModel(opt=opt,
                                          num_features=len(self.dict),
                                          padding_idx=self.NULL_IDX,
                                          start_idx=self.START_IDX,
                                          end_idx=self.END_IDX,
                                          longest_label=states.get('longest_label', 1))
            elif ARCH_CHOICE == 'gpt':
                assert isinstance(self.dict, GPTDictionaryAgent)
                self.model = Gpt2SeqModel(opt=opt,
                                          vocab_size=len(self.dict),
                                          pad_idx=self.NULL_IDX,
                                          start_idx=self.START_IDX,
                                          end_idx=self.END_IDX,
                                          dict=self.dict,
                                          special_token_len=len(self.dict.special_tokens),
                                          longest_label=states.get('longest_label', 1))

            if opt.get('display_model', False):
                print_model(self.model)

            if opt.get('dict_tokenizer') == 'bpe' and opt['embedding_type'] != 'random':
                print('skipping preinitialization of embeddings for bpe')

            elif not states and opt['embedding_type'] != 'random' and ARCH_CHOICE == 'lstm':
                # set up preinitialized embeddings
                try:
                    import torchtext.vocab as vocab
                except ImportError as ex:
                    print('Please install torch text with `pip install torchtext`')
                    raise ex
                pretrained_dim = 300
                if opt['embedding_type'].startswith('glove'):
                    if 'twitter' in opt['embedding_type']:
                        init = 'glove-twitter'
                        name = 'twitter.27B'
                        pretrained_dim = 200
                    else:
                        init = 'glove'
                        name = '840B'
                    embs = vocab.GloVe(name=name, dim=pretrained_dim,
                                       cache=modelzoo_path(self.opt.get('datapath'),
                                                           'models:glove_vectors')
                                       )
                elif opt['embedding_type'].startswith('fasttext'):
                    init = 'fasttext'
                    embs = vocab.FastText(language='en',
                                          cache=modelzoo_path(self.opt.get('datapath'),
                                                              'models:fasttext_vectors')
                                          )
                else:
                    raise RuntimeError('embedding type not implemented')

                if opt['encoder_embed_dim'] != pretrained_dim:
                    rp = torch.Tensor(pretrained_dim, opt['encoder_embed_dim']).normal_()
                    t = lambda x: torch.mm(x.unsqueeze(0), rp)
                else:
                    t = lambda x: x
                cnt = 0
                for w, i in self.dict.tok2ind.items():
                    if w in embs.stoi:
                        vec = t(embs.vectors[embs.stoi[w]])
                        self.model.decoder.tgt_word_emb.weight.data[i] = vec
                        cnt += 1
                        if opt['lookuptable'] in ['unique', 'dec_out']:
                            # also set encoder lt, since it's not shared
                            self.model.encoder.src_word_emb.weight.data[i] = vec
                print('Seq2seq: initialized embeddings for {} tokens from {}.'
                      ''.format(cnt, init))

            if states:
                # set loaded states if applicable
                self.model.load_state_dict(states['model'])

            if self.use_cuda:
                self.model.cuda()

            # if select persona
            if opt['select_persona']:
                self.receiver, self.receiver_dict = self.load_receiver(opt['receiver_model'])
                self.receiver.eval()
                # move to cuda
                self.receiver.cuda()
            else:
                self.receiver = None
                self.receiver_dict = None

        vocab_size = len(self.dict.tok2ind.items())

        if opt['smoothing'] > 0.0:
            self.criterion = LabelSmoothingLoss(vocabulary_size=40516,
                                                label_smoothing=opt['smoothing'],
                                                pad_index=self.NULL_IDX)
        else:
            self.criterion = TokenCrossEntropyLoss(pad_index=self.NULL_IDX)

        self.class_criter = nn.CrossEntropyLoss()
        self.eval_criterion = TokenCrossEntropyLoss(pad_index=self.NULL_IDX)
        # whether shuffle persona
        self.shuffle_persona = opt['shuffle_persona']

        if self.use_cuda:
            self.criterion.cuda()

        if 'train' in opt.get('datatype', ''):
            # we only set up optimizers when training
            # we only set this up for the original instance or hogwild ones
            self.clip = opt.get('gradient_clip', -1)

            # set up optimizer
            lr = opt['lr']
            optim_class = TransformerAgent.OPTIM_OPTS[opt['optimizer']]
            if ARCH_CHOICE == 'lstm':
                kwargs = {'lr': lr}
                if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop']:
                    kwargs['momentum'] = opt['momentum']
                    if opt['optimizer'] == 'sgd':
                        kwargs['nesterov'] = True
                if opt['optimizer'] == 'adam':
                    kwargs['amsgrad'] = True

                if opt['embedding_type'].endswith('fixed'):
                    print('Transformer: fixing embedding weights.')
                    self.model.decoder.tgt_word_emb.weight.requires_grad = False
                    self.model.encoder.src_word_emb.weight.requires_grad = False

                    if opt['lookuptable'] in ['dec_out', 'all']:
                        self.model.decoder.e2s.weight.requires_grad = False
                self.optimizer = optim_class([p for p in self.model.parameters() if p.requires_grad], **kwargs)
            elif ARCH_CHOICE == 'gpt':
                self.optimizer = GPTOptimizer(self.model, opt)

            if states.get('optimizer'):
                if states['optimizer_type'] != opt['optimizer']:
                    print('WARNING: not loading optim state since optim class '
                          'changed.')
                else:
                    try:
                        self.optimizer.load_state_dict(states['optimizer'])
                    except ValueError:
                        print('WARNING: not loading optim state since model '
                              'params changed.')
                    # if self.use_cuda:
                    #     for state in self.optimizer.state.values():
                    #         for k, v in state.items():
                    #             if isinstance(v, torch.Tensor):
                    #                 state[k] = v.cuda()
            if ARCH_CHOICE == 'lstm':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 'min', factor=0.5, patience=3, verbose=True)

        self.step = torch.zeros(1)
        self.reset()

    @staticmethod
    def load_receiver(receiver_model_file):
        from parlai.core.agents import create_agent
        # option file read
        optfile = receiver_model_file + '.opt'
        with open(optfile, 'rb') as handle:
            opt = pickle.load(handle)
        receiver_agent = create_agent(opt)
        # load states
        status = torch.load(receiver_model_file, map_location=lambda cpu, _: cpu)
        receiver_dict = receiver_agent.dict
        model = receiver_agent.encoder
        model.load_state_dict(status['model'])
        return model, receiver_dict

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'optimizer', 'lookuptable', 'beam_size'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('[ Adding new option: | {k}: {v} | ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('[ Overriding option: | {k}: {old} => {v} | ]'.format(
                    k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        if 'dict_file' in new_opt and not self.opt.get('dict_file'):
            print('[ No dictionary path detected, trying to load previous '
                  'path {} ]'.format(new_opt['dict_file']))
            self.opt['dict_file'] = new_opt['dict_file']
        return self.opt

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        new_vec = []
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.history.clear()
        for i in range(len(self.answers)):
            self.answers[i] = None
        self.reset_metrics()

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0
        self.metrics['correct_pred'] = 0
        self.metrics['pred_count'] = 0

    def report(self):
        """Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            m['loss'] = self.metrics['loss'] / num_tok
            if self.metrics['pred_count'] > 0:
                m['pred'] = self.metrics['correct_pred'] / self.metrics['pred_count']
            try:
                m['ppl'] = math.exp(m['loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        if self.metrics['total_skipped_batches'] > 0:
            m['total_skipped_batches'] = self.metrics['total_skipped_batches']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['opt'] = self.opt
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        shared['START_IDX'] = self.START_IDX
        shared['END_IDX'] = self.END_IDX
        shared['NULL_IDX'] = self.NULL_IDX
        shared['model'] = self.model
        shared['receiver'] = self.receiver
        shared['receiver_dict'] = self.receiver_dict
        if self.opt.get('numthreads', 1) > 1:
            # we're doing hogwild so share the model too
            if type(self.metrics) == dict:
                # move metrics and model to shared memory
                self.metrics = SharedTable(self.metrics)
                self.model.share_memory()
            shared['states'] = {  # don't share optimizer states
                'optimizer_type': self.opt['optimizer'],
            }
        shared['metrics'] = self.metrics  # do after numthreads check
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        is_training = 'labels' in observation

        if not obs.get('preprocessed', False) or 'text2vec' not in obs:
            # preprocess text in observations, replace candidates persona into another tagging
            if obs.get('text', False):
                # must keep lower to avoid unk
                text_split = obs['text'].lower().split('\n')
                persona_given = ''
                for t in text_split:
                    if t.startswith('your persona') or t.startswith('their persona'):
                        t = t.replace('your persona: ', '').replace('their persona: ', '')
                        persona_given += t + '\n'
                # TODO: drop the old method to encode persona, we use the separate encoder to do that.
                if self.use_talk_tokens:
                    obs['text'] = SpecialToken.talk_1_start + ' ' + text_split[-1] + ' ' + SpecialToken.talk_1_end
                else:
                    obs['text'] = text_split[-1]

                obs['persona'] = persona_given

            obs['text2vec'], obs['dis2vec'], obs['turn2vec'], obs['cur_turn'] = maintain_dialog_history(
                self.history, obs,
                reply=self.answers[self.batch_idx],
                max_history_len=self.encode_max_seq_len,
                use_reply=self.use_history_reply,
                persona_append_strategy=self.persona_append_strategy,
                history_append_strategy=self.history_append_strategy,
                receiver=self.receiver,
                receiver_dict=self.receiver_dict,
                use_persona_tokens=self.use_person_tokens,
                shuffle_persona=self.shuffle_persona and is_training,
                dict=self.dict)
        else:
            obs['text2vec'] = deque(obs['text2vec'], maxlen=self.encode_max_seq_len)
            obs['dis2vec'] = deque(obs['dis2vec'], maxlen=self.encode_max_seq_len)
            obs['turn2vec'] = deque(obs['turn2vec'], maxlen=self.encode_max_seq_len)

        self.observation = obs
        self.answers[self.batch_idx] = None
        return obs

    def predict(self, src_seq, src_seq_turn, src_seq_dis, tgt_seq=None, tgt_seq_turn=None, cands=None, valid_cands=None,
                sampling_cands=None, is_training=False):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        predictions, cand_preds = None, None
        if is_training:
            self.model.train()
            self.zero_grad()
            out = None
            try:
                # keep the same interface
                out = self.model.forward(src_seq=src_seq,
                                         src_seq_turn=src_seq_turn,
                                         src_seq_dis=src_seq_dis,
                                         tgt_seq=tgt_seq,
                                         tgt_seq_turn=tgt_seq_turn,
                                         rank_during_training=False,
                                         cands=cands,
                                         sampling_cands=sampling_cands,
                                         valid_cands=valid_cands)
                # generated response
                _preds, scores, cand_preds = out[0], out[1], out[2]
                positive_score, negative_score = out[-2], out[-1]

                positive_pred = torch.argmax(positive_score, dim=1)
                negative_pred = torch.argmax(negative_score, dim=1)
                rank_correct = positive_pred.ne(0).sum() + negative_pred.ne(1).sum()

                y_ne = tgt_seq.ne(self.NULL_IDX)
                target_tokens = y_ne.long().sum().item()
                correct = ((tgt_seq == _preds) * y_ne).sum().item()
                pos_label = torch.tensor([1] * positive_score.size(0), device=positive_score.device)
                neg_label = torch.tensor([0] * negative_score.size(0), device=positive_score.device)

                gen_loss = self.criterion(scores, tgt_seq) / target_tokens
                class_loss = (self.class_criter(positive_score, pos_label) + self.class_criter(negative_score, neg_label))/2
                loss = 0.6 * gen_loss + 0.4 * class_loss
                # save loss to metrics

                self.metrics['correct_tokens'] += correct
                self.metrics['loss'] += gen_loss.item()
                self.metrics['num_tokens'] += 1
                self.metrics['correct_pred'] += rank_correct.item()
                self.metrics['pred_count'] += positive_score.size(0) * 2
                loss.backward()
            except RuntimeError as e:
                # catch out of memory exceptions during fwd/bck (skip batch)
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch. '
                          'if this happens frequently, decrease batchsize or '
                          'truncate the inputs to the model.')
                    self.metrics['total_skipped_batches'] += 1
                    return predictions, cand_preds
                else:
                    raise e
            self.update_params()
        else:
            self.model.eval()
            out = self.model.forward(src_seq=src_seq,
                                     src_seq_turn=src_seq_turn,
                                     src_seq_dis=src_seq_dis,
                                     rank_during_training=cands is not None,
                                     cands=cands,
                                     valid_cands=valid_cands)
            predictions, cand_preds = out[0], out[2]

            if tgt_seq is not None:
                # calculate loss on targets
                out = self.model.forward(src_seq=src_seq,
                                         src_seq_turn=src_seq_turn,
                                         src_seq_dis=src_seq_dis,
                                         tgt_seq=tgt_seq,
                                         tgt_seq_turn=tgt_seq_turn,
                                         cands=cands,
                                         valid_cands=valid_cands)
                scores = out[1]
                # just used to calculate perplexity
                with torch.no_grad():
                    loss = self.eval_criterion(scores, tgt_seq)
                # save loss to metrics
                target_tokens = tgt_seq.ne(self.NULL_IDX).long().sum().item()
                self.metrics['loss'] += loss.item()
                self.metrics['num_tokens'] += target_tokens

        return predictions, cand_preds

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""
        is_training = any(['labels' in obs for obs in observations])

        src_seq, tgt_seq, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, end_idx=self.END_IDX,
            null_idx=self.NULL_IDX, dq=True, eval_labels=True,
            encode_truncate=self.encode_max_seq_len, decode_truncate=self.decode_max_seq_len)

        max_seq_len = len(src_seq[0])
        # now the source sequence turn is just `relative distance`
        src_seq_dis = []
        # TODO: add turn embedding for src_seq
        for cur_ind, org_ind in enumerate(valid_inds):
            org_dis_ids = observations[org_ind]['dis2vec'].copy()
            org_dis_ids.extend([0] * (max_seq_len - len(org_dis_ids)))
            src_seq_dis.append(org_dis_ids)

        src_seq_turn = []
        tgt_seq_turn = []
        # TODO: add turn embedding for src_seq
        for cur_ind, org_ind in enumerate(valid_inds):
            org_turn_ids = observations[org_ind]['turn2vec'].copy()
            org_turn_ids.extend([0] * (max_seq_len - len(org_turn_ids)))
            src_seq_turn.append(org_turn_ids)
            # decode turn id as input
            tgt_seq_turn.append(observations[org_ind]['cur_turn'])

        if src_seq is None:
            return None, None, None, None, None, None, None

        src_seq = torch.LongTensor(src_seq)
        # src_seq_turn = torch.LongTensor(src_seq_turn)
        # src_seq_dis = torch.LongTensor(src_seq_dis)
        # tgt_seq_turn = torch.LongTensor(tgt_seq_turn)

        if tgt_seq is not None:
            tgt_seq = torch.LongTensor(tgt_seq)

        if self.use_cuda:
            # copy to gpu
            src_seq = src_seq.cuda()
            # src_seq_turn = src_seq_turn.cuda()
            # src_seq_dis = src_seq_dis.cuda()
            # tgt_seq_turn = tgt_seq_turn.cuda()
            if tgt_seq is not None:
                tgt_seq = tgt_seq.cuda()

        # set up candidates
        cands = []
        sampling_cands = []
        valid_cands = []
        for i, v in enumerate(valid_inds):
            if 'label_candidates' in observations[v]:
                curr_lcs = list(observations[v]['label_candidates'])
                curr_cands = [{'text': c + ' ' + self.dict.end_token} for c in curr_lcs]
                # padding candidates
                cs, _, _, valid_c_inds, *_ = PaddingUtils.pad_text(curr_cands, self.dict, null_idx=self.NULL_IDX,
                                                                   # TODO: whether add end idx to add
                                                                   dq=True, encode_truncate=self.decode_max_seq_len)
                valid_cands.append((i, v, [curr_lcs[j] for j in valid_c_inds]))
                cs = torch.LongTensor(cs)
                if self.use_cuda:
                    cs = cs.cuda()
                cands.append(cs)
                # random select one from 0:18 from curr_lcs
                sampling_cands.append(random.choice(curr_lcs[:19]) + ' ' + self.dict.end_token)
        # construct one tensor
        sample_can_sep = ' {} '.format(self.dict.start_token).join(sampling_cands)
        # the sample should appended a END symbol as well.
        sample_out = PaddingUtils.pad_text([{'text': sample_can_sep, 'eval_labels': [sample_can_sep]}],
                                           self.dict, null_idx=self.NULL_IDX, dq=False)
        # remove the last which is extra END IDX
        sample_ys = sample_out[1]
        sampling_cands = split_pad_vector(sample_ys, self.START_IDX, self.NULL_IDX)[0]
        sampling_cands = torch.LongTensor(sampling_cands)
        if self.use_cuda:
            sampling_cands = sampling_cands.cuda()

        return src_seq, src_seq_turn, src_seq_dis, tgt_seq, tgt_seq_turn, labels, valid_inds, cands, valid_cands, sampling_cands, is_training

    def init_cuda_buffer(self, batchsize):
        if self.use_cuda and not hasattr(self, 'buffer_initialized'):
            try:
                print('preinitializing pytorch cuda buffer')
                bsz = self.opt.get('batchsize', batchsize)
                input_dummy = torch.ones(bsz, self.encode_max_seq_len).long().cuda()
                output_dummy = torch.ones(bsz, 1).long().cuda()
                sc = self.model(input_dummy, None, None, output_dummy, None)[1]
                if self.opt['datatype'] == 'train':
                    loss = self.criterion(sc, output_dummy)
                    loss.backward()
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = ('CUDA OOM: Lower batch size (-bs) from {} or lower max'
                         ' sequence length (-tr) from {}'.format(bsz, self.encode_max_seq_len))
                    raise RuntimeError(m)
                else:
                    raise e

    def batch_act(self, observations):
        batchsize = len(observations)
        self.init_cuda_buffer(batchsize)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        src_seq, src_seq_turn, src_seq_dis, tgt_seq, tgt_seq_turn, labels, valid_inds, cands, valid_cands, sampling_cands, is_training = self.vectorize(
            observations)

        if src_seq is None:
            # no valid examples, just return empty responses
            return batch_reply

        # produce best_pred, train on targets if availables
        cand_inds = [i[0] for i in valid_cands] if valid_cands is not None else None
        predictions, cand_preds = self.predict(src_seq, src_seq_turn, src_seq_dis, tgt_seq, tgt_seq_turn, cands,
                                               cand_inds, sampling_cands, is_training)

        if is_training:
            report_freq = 0
        else:
            report_freq = self.report_freq
        if predictions is not None:
            PaddingUtils.map_predictions(
                predictions, valid_inds, batch_reply, observations,
                self.dict, self.END_IDX, report_freq=report_freq, labels=labels,
                answers=self.answers, ys=tgt_seq.data if tgt_seq is not None else None)

        if cand_preds is not None:
            if valid_cands is None:
                valid_cands = [(None, i, labels) for i in valid_inds]
            for i in range(len(valid_cands)):
                order = cand_preds[i]
                _, batch_idx, curr_cands = valid_cands[i]
                curr = batch_reply[batch_idx]
                curr['text_candidates'] = [curr_cands[idx] for idx in order
                                           if idx < len(curr_cands)]

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {'model': self.model.state_dict(),
                     'longest_label': self.model.longest_label,
                     'optimizer': self.optimizer.state_dict(),
                     'optimizer_type': self.opt['optimizer']}

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file as json
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None and hasattr(self, 'optimizer'):
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        return states

    def receive_metrics(self, metrics_dict):
        """Use the metrics to decide when to adjust LR schedule."""
        if 'loss' in metrics_dict and ARCH_CHOICE == 'lstm':
            self.scheduler.step(metrics_dict['loss'])


class mydefaultdict(defaultdict):
    """Custom defaultdict which overrides defaults requested by the get
    function with the default factory.
    """

    def get(self, key, default=None):
        # override default from "get" (like "__getitem__" already is)
        return super().get(key, default or self.default_factory())
