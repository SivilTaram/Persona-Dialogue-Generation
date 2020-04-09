import copy
import math
import os
import pickle
import random
import numpy as np
import torch
from parlai.agents.seq2seq.modules import RNNEncoder as Encoder
from parlai.core.agents import Agent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils, round_sigfigs
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch import optim
import json
from agents.common.bert_dictonary import BertDictionaryAgent
from agents.common.bert_optimizer import get_bert_optimizer
import torch.nn.functional as F


def _calculate_expect_softmax_loss(score_view, ys):
    denominator = np.exp(score_view).sum(-1)
    numerator = []
    for i, y in enumerate(ys):
        numerator.append(np.exp(score_view[i][y]).reshape(1, ))
    numerator = np.concatenate(numerator, axis=0)
    out = numerator / denominator
    return np.mean(- np.log(out))


def padding(_xs, max_1d_size, max_2d_size, null_idx):
    for _ in range(max_1d_size - len(_xs)):
        _xs.append([])
    for x in _xs:
        for _ in range(max_2d_size - len(x)):
            x.append(null_idx)


def padding4d(_xs, max_1d_size, max_2d_size, max_3d_size, null_idx):
    for _ in range(max_1d_size - len(_xs)):
        _xs.append([])
    for x in _xs:
        for _ in range(max_2d_size - len(x)):
            x.append([])
    for x in _xs:
        for y in x:
            for _ in range(max_3d_size - len(y)):
                y.append(null_idx)


def split_pad_vector(xs, separator, null_idx):
    """
    Use the splitor to split the sentences.

    spliter is the value that represents END TOKEN
    :param x: input
    :param separator: the required seperator
    :return: a list of dialogs after splitting and padding
    """

    def split(x):
        _xs = []
        temp_x = []
        for _x in x:
            if _x == separator:
                _xs.append(temp_x)
                temp_x = []
                continue
            if _x != null_idx:
                temp_x.append(_x)
        if len(temp_x):
            _xs.append(temp_x)
        return _xs

    def get_max_words_size(_xs):
        max_size = 0
        for agent in _xs:
            for dialog in agent:
                if len(dialog) > max_size:
                    max_size = len(dialog)
        return max_size

    xs = [split(x) for x in xs]
    max_turn_size = max((len(x) for x in xs))
    max_words_size = get_max_words_size(xs)
    for agent in xs:
        padding(agent, max_turn_size, max_words_size, null_idx)
    return xs


class ReceiverAgent(Agent):
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
    }
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/features/weights/opts from this file')
        agent.add_argument('-hs', '--sent-hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-des', '--dialog-embedding-size', type=int, default=128,
                           help='size of the dialog embedding')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-lr', '--learningrate', type=float, default=1,
                           help='learning rate')
        agent.add_argument('-wd', '--weight_decay', type=float, default=1e-5,
                           help='L2 normalization')
        agent.add_argument('-blr', '--bert_learning_rate', type=float, default=5e-5,
                           help='BERT model learning rate')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-indr', '--input_dropout', type=float, default=0.1,
                           help='embedding dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'general'],
                           help='Attention type')
        agent.add_argument('--no-cuda', action='store_true', default=False,
                           help='disable GPUs even if available')
        agent.add_argument('-gpu', '--gpu', type=int, default=0,
                           help='which GPU device to use')
        # ranking arguments
        agent.add_argument('-rc', '--rank-candidates', type='bool',
                           default=False,
                           help='rank candidates if available. this is done by'
                                ' computing the prob score per token for each '
                                'candidate and selecting the highest scoring.')
        agent.add_argument('-tr', '--truncate', type=int, default=-1,
                           help='truncate input & output lengths to speed up '
                                'training (may reduce accuracy). This fixes all '
                                'input and output to have a maximum length. This '
                                'reduces the total amount '
                                'of padding in the batches.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=ReceiverAgent.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-opt', '--optimizer', default='sgd',
                           choices=ReceiverAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-mom', '--momentum', default=0.9, type=float,
                           help='if applicable, momentum value for optimizer. '
                                'if > 0, sgd uses nesterov momentum.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-rf', '--report-freq', type=float, default=0.001,
                           help='Report frequency of prediction during eval.')
        agent.add_argument('-histr', '--history-replies',
                           default='label_else_model', type=str,
                           choices=['none', 'model', 'label',
                                    'label_else_model'],
                           help='Keep replies in the history, or not.')
        agent.add_argument('-lfcp', '--load-from-checkpoint', type='bool', default=False,
                           help='load checkpoint model')
        agent.add_argument('-dictbuildfirst', '--dict-build-first', type='bool', default=False,
                           help='build dict before loading module')
        agent.add_argument('-crt', '--criterion', type=str, default='cross_entropy',
                           choices=['cross_entropy', 'margin'],
                           help='build dict before loading module')
        agent.add_argument('--marginloss-margin', type=float, default=1.0,
                           help='The margin parameter in marginloss')
        agent.add_argument('-emb', '--embedding-type', default='random',
                           choices=['random', 'glove', 'glove-fixed'],
                           help='Choose between different strategies '
                                'for word embeddings. Default is random, '
                                'but can also preinitialize from Glove or '
                                'Fasttext.'
                                'Preinitialized embeddings can also be fixed '
                                'so they are not updated during training.')
        # bert arugments
        agent.add_argument('-bpp', '--bert_pretrained_path', default='bert-base-uncased')
        ReceiverAgent.dictionary_class().add_cmdline_args(argparser)

        agent.add_argument('--sparse', default=5e-2, help='sparse constraints')

        # score arguments
        agent.add_argument('--score_train', type=str, choices=['sum', 'inc'])
        agent.add_argument('--score_normalize', type=str, choices=['scale', 'sigmoid'])
        agent.add_argument('--score_method', type=str, choices=['dot', 'bilinear', 'cos'])

        return agent

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.use_cuda = not opt.get('no_cuda') and torch.cuda.is_available()

        self.sparse = opt.get('sparse', 0.0)
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.observation = None

        self.answers = [None] * opt['batchsize']
        self.report_freq = opt.get('report_freq', 0.001)
        if opt['criterion'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif opt['criterion'] == 'margin':
            self.criterion = nn.MultiMarginLoss(margin=opt['marginloss_margin'])
        elif opt['criterion'] == 'cyh':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        # WARNING on learning rate
        self.lr = opt['learningrate']
        self.bert_lr = opt['bert_learning_rate']
        if self.bert_lr >= 1e-4:
            print(
                '[ Warning: You should fine-tune the BERT using a smaller learning rate, current setting is {} ]'.format(
                    self.bert_lr))

        self.dict = None

        if shared is None:
            self.metrics = {
                'loss': 0.0,
                'num_cands': 0,
                'num_correct': 0,
                'num_exs': 0,
                'positive_score_sum': 0.0,
                'score_sum': 0.0,
                'score_std': 0.0,
                'norm_hit@1': 0.0,
                'mean_avg_pre': 0.0,
            }
            self.__init_dict()
            self.encoder = ReceiverEncoder(opt, len(self.dict),
                                           self.NULL_IDX, self.START_IDX, self.END_IDX, self.UNK_IDX)

            # CUDA should be called before the construction of optimizer
            # elif the Adam optimizer will create a tensor in a different gpu
            if self.use_cuda:
                self.encoder.cuda()

            self.optimizer = get_bert_optimizer(models=[self.encoder],
                                                bert_learning_rate=self.bert_lr,
                                                base_learning_rate=self.lr,
                                                weight_decay=self.opt['weight_decay'])

            self.lr_decay = [1]
            self.__init_model()
        else:
            self.metrics = shared['metrics']
            self.dict = shared['dict']
            self.__init_dict()
            self.encoder = shared['encoder']
            self.optimizer = shared['optimizer']
            self.lr_decay = shared['lr_decay']

    def share(self):
        shared = super().share()
        shared['dict'] = self.dict
        shared['metrics'] = self.metrics
        shared['encoder'] = self.encoder
        shared['optimizer'] = self.optimizer
        shared['lr_decay'] = self.lr_decay
        return shared

    def __init_dict(self):
        if self.dict is None:
            self.dict = self.dictionary_class()(self.opt)
        self.START_IDX = self.dict[self.dict.start_token]
        # we use END markers to end our output
        self.END_IDX = self.dict[self.dict.end_token]
        # get index of null token from dictionary (probably 0)
        self.NULL_IDX = self.dict[self.dict.null_token]
        self.UNK_IDX = self.dict[self.dict.unk_token]

    def __init_model(self):
        opt = self.opt
        init_model = None
        states = None
        # check first for 'init_model' for loading model from file
        if opt.get('init_model') and os.path.isfile(opt['init_model']):
            init_model = opt['init_model']
        # next check for 'model_file', this would override init_model
        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            init_model = opt['model_file']
        if init_model \
                and opt.get('load_from_checkpoint') \
                and os.path.isfile(init_model + '.checkpoint'):
            init_model += '.checkpoint'

        if init_model is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'.format(init_model))
            states = self.load(init_model)

            if os.path.isfile(init_model + '.dict') or opt['dict_file'] is None:
                opt['dict_file'] = init_model + '.dict'

        if states:
            self.__load_model_from_states(states)

    def __load_embedding(self):
        opt = self.opt
        # set up preinitialized embeddings
        try:
            import torchtext.vocab as vocab
        except ImportError as ex:
            print('Please install torch text with `pip install torchtext`')
            raise ex
        pretrained_dim = 300
        init = ''
        if opt['embedding_type'].startswith('glove'):
            init = 'glove'
            name = '840B'
            embs = vocab.GloVe(name=name, dim=pretrained_dim,
                               cache=modelzoo_path(self.opt.get('datapath'),
                                                   'models:glove_vectors')
                               )
        else:
            raise RuntimeError('embedding type not implemented')

        if opt['embeddingsize'] != pretrained_dim:
            rp = torch.Tensor(pretrained_dim, opt['embeddingsize']).normal_()
            t = lambda x: torch.mm(x.unsqueeze(0), rp)
        else:
            t = lambda x: x
        cnt = 0
        for w, i in self.dict.tok2ind.items():
            if w in embs.stoi:
                vec = t(embs.vectors[embs.stoi[w]])
                self.encoder.sent_encoder.lt.weight.data[i] = vec
                self.encoder.persona_encoder.lt.weight.data[i] = vec
                cnt += 1
        print('D2Receiver: initialized embeddings for {} tokens from {}.'
              ''.format(cnt, init))

    def __load_model_from_states(self, states):
        opt = self.opt
        self.encoder.load_state_dict(states['model'], strict=True)
        if not states.get('optimizer'):
            return

        if states['optimizer_type'] != opt['optimizer']:
            print('WARNING: not loading optim state since optim class '
                  'changed.')
        else:
            try:
                self.optimizer.load_state_dict(states['optimizer'])
            except ValueError:
                print('WARNING: not loading optim state since model '
                      'params changed.')
            if self.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    def __setup_optimizer(self):
        opt = self.opt
        optim_class = ReceiverAgent.OPTIM_OPTS[opt['optimizer']]
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop']:
            kwargs['momentum'] = opt['momentum']
            if opt['optimizer'] == 'sgd':
                kwargs['nesterov'] = True
        if opt['optimizer'] == 'adam':
            # https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True

        # L2 regularization to avoid overfit
        kwargs['weight_decay'] = 1e-5
        return optim_class([p for p in self.encoder.dialog_encoder.parameters() if p.requires_grad], **kwargs)

    def observe(self, observation):
        obs = copy.deepcopy(observation)
        self.observation = obs
        return obs

    def vectorize(self, observations):

        """Convert a list of observations into input & target tensors."""
        xs, _, labels, valid_inds, _, _ = PaddingUtils.pad_text(
            observations, self.dict, end_idx=self.END_IDX,
            null_idx=self.NULL_IDX, dq=True, eval_labels=True,
            truncate=self.truncate)

        if xs is None:
            raise RuntimeError()

        xs = split_pad_vector(xs, self.END_IDX, self.NULL_IDX)

        xs = torch.LongTensor(xs)
        cands = []
        valid_cands = []
        ys = []
        all_cands = []
        max_2d_size = 0
        max_3d_size = 0
        max_4d_size = 0
        for xs_index, original_index in enumerate(valid_inds):
            if 'label_candidates' in observations[original_index]:
                obs = observations[original_index]
                candidates = obs['label_candidates']
                curr_lcs = list(obs['label_candidates'])
                curr_cands = [{'text': c} for c in curr_lcs]
                cs, _, _, valid_c_inds, *_ = PaddingUtils.pad_text(curr_cands, self.dict, null_idx=self.NULL_IDX,
                                                                   dq=True, truncate=self.truncate)
                cs = split_pad_vector(cs, self.END_IDX, self.NULL_IDX)
                # reindex correct answer position
                label_kw = 'labels' if 'labels' in obs else 'eval_labels'
                ans = obs[label_kw][0]

                all_ans = [candidates.index(s) for s in obs[label_kw]]
                all_ans = [valid_c_inds.index(ans_ind) for ans_ind in all_ans]
                all_cands.append(all_ans)

                y_index = candidates.index(ans)
                valid_cands.append((xs_index, original_index, valid_c_inds, [curr_lcs[j] for j in valid_c_inds]))
                cands.append(cs)
                ys.append(valid_c_inds.index(y_index))
                # assert valid_c_inds.index(0) == ys[xs_index], "Correct answer should be in the index of 0, " \
                #                                               "and saved to observation['labels']"
                if len(cs) > max_2d_size:
                    max_2d_size = len(cs)
                if len(cs[0]) > max_3d_size:
                    max_3d_size = len(cs[0])
                if len(cs[0][0]) > max_4d_size:
                    max_4d_size = len(cs[0][0])
            else:
                raise NotImplemented()
        for cand in cands:
            padding4d(cand,
                      max_1d_size=max_2d_size,
                      max_2d_size=max_3d_size,
                      max_3d_size=max_4d_size,
                      null_idx=self.NULL_IDX)
        ys = torch.LongTensor(ys)
        # num_batch x num_candidate x num_persona x persona_token
        cands = torch.LongTensor(cands)

        if self.use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            cands = cands.cuda()

        return xs, ys, labels, valid_inds, cands, valid_cands, all_cands

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        is_training = any(['labels' in obs for obs in observations])

        if is_training:
            xs, ys, labels, valid_inds, cands, valid_cands, all_ans = self.vectorize(observations)
        else:
            with torch.no_grad():
                xs, ys, labels, valid_inds, cands, valid_cands, all_ans = self.vectorize(observations)

        if xs is None:
            # no valid examples, just return empty responses
            return batch_reply

        if is_training:
            predictions, cand_preds, grid_score, grid_mask = self.predict(xs, ys, cands, is_training, all_ans)
        else:
            with torch.no_grad():
                # produce predictions, train on targets if availables
                predictions, cand_preds, grid_score, grid_mask = self.predict(xs, ys, cands, is_training, all_ans)

        if is_training:
            report_freq = 0
        else:
            report_freq = self.report_freq

        self.map_predictions(
            cand_preds, valid_cands, valid_inds, batch_reply, observations,
            self.dict, self.END_IDX, report_freq=report_freq, labels=labels,
            grid_score=grid_score, grid_mask=grid_mask,
            answers=self.answers, ys=ys.data if ys is not None else None,
            cands=cands)

        return batch_reply

    def map_predictions(self, predictions, valid_cands, valid_inds, batch_reply, observations,
                        dictionary, end_idx, report_freq=0.1, grid_score=None, grid_mask=None,
                        labels=None, answers=None, ys=None, cands=None):
        """Predictions are mapped back to appropriate indices in the batch_reply
           using valid_inds.
           report_freq -- how often we report predictions

           This method will modify batch_reply directly
        """

        def _remove_null_token(_input):
            """
            remove the ending __NULL__ token from the data.

            otherwise the agent replay will contain unwanted __NULL__ tokens,
            which will affect the accuracy result (calculated by Metrics.update
            method, requiring exact match)
            """
            end_index = len(_input) - 1
            for end_index in range(len(_input) - 1, -1, -1):
                if _input[end_index] != self.NULL_IDX:
                    break
            return _input[:end_index + 1]

        def _token2txt(_tokens):
            _tokens = _remove_null_token(_tokens)
            return dictionary.vec2txt(_tokens)

        def _rank_cands(_cands, _pred):
            _cands = copy.deepcopy(_cands)
            ans = sorted(zip(_cands, _pred), key=lambda x: -x[1])
            ans = list(zip(*ans))[0]
            return ans

        def _get_order_cands_txt(_cands, _pred):
            return list(map(_token2txt,
                            _rank_cands(_cands, _pred)))

        if self.use_cuda:
            predictions = predictions.cpu()

        # transform cands into actual tokens
        cand_shape = cands.size()
        # num_batch x num_candidate x persona
        cands = cands.view(cand_shape[0], cand_shape[1], -1)

        # TODO: very slow. so remove it to speed up the validation
        # cands = _remove_start_end_padding(cands)
        predictions = predictions.detach().numpy()
        for xs_index in range(len(predictions)):
            original_index = valid_inds[xs_index]
            # map the predictions back to non-empty examples in the batch
            # we join with spaces since we produce tokens one at a timelab
            curr = batch_reply[original_index]
            output_tokens = cands[xs_index][predictions[xs_index].argmax()]
            # flatten output tokens
            curr_pred = _token2txt(output_tokens)

            no_sort_candidates = valid_cands[xs_index][-1]

            # enable hit@k indicator
            # WARNING: BERT use word piece not whole word.
            # To be more robust, here should use the labels[xs_index]
            if labels is not None:
                # get the actual index in xs_index
                order_list = np.argsort(-predictions[xs_index])
                curr['text_candidates'] = [no_sort_candidates[ind] for ind in order_list]
                curr['text'] = curr['text_candidates'][0]
            else:
                curr['text_candidates'] = _get_order_cands_txt(cands[xs_index], predictions[xs_index])
                curr['text'] = curr_pred

            if labels is not None and answers is not None:
                answers[original_index] = labels[xs_index]
            elif answers is not None:
                answers[original_index] = output_tokens

            if random.random() > (1 - report_freq):
                # log sometimes
                print('TEXT: ', observations[original_index]['text'])
                print('PREDICTION: ', curr_pred, '\n~')
        return

    def act(self):
        return self.batch_act([self.observation])[0]

    def reset(self):
        self.observation = None
        for i in range(len(self.answers)):
            self.answers[i] = None
        self.metrics['loss'] = 0.0
        self.metrics['num_cands'] = 0
        self.metrics['num_correct'] = 0
        self.metrics['num_exs'] = 0
        self.metrics['positive_score_sum'] = 0.0
        self.metrics['score_sum'] = 0.0
        self.metrics['score_std'] = 0.0
        self.metrics['mean_avg_pre'] = 0.0
        if 'norm_hit@1' in self.metrics:
            self.metrics['norm_hit@1'] = 0.0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        self.optimizer.step()

    def encode_personae(self, personae):
        """

        :param personae, a list of string that needs embedding
        :return: tensor with size [num_persona, num_persona_embedding_size]
        """
        return self.encoder.encode_persona(personae, self.dict, use_cuda=self.use_cuda)

    def encode_dialogs(self, dialogs):
        """

        :param dialogs: a list of dialog that need converting

            it should follow the format as:

            ```
            a0 __START__ b0 __END__ a1 __START__ b1 __END___ ...

            ```
        Note: The part between __START__ and ___END__ corresponds to texts which describes
              the target persona
        :return: embeded dialogs with size [num_dialogs, num_embedding_size]
        """
        return self.encoder.encode_dialog(dialogs, self.dict, use_cuda=self.use_cuda)

    def predict(self, xs, ys, cands, is_training=False, all_answers=None):
        """

        :param xs:
        :param ys: index of true corresponding persona
        :param cands: candidates contain all negative sampling persona and true persona. The receiver aims to convert
        utterances into personas, the margin loss could improve the positive/negative distinction.
        :param is_training:
        :return:
        """
        if is_training:
            self.encoder.train()
            self.zero_grad()
        else:
            # this will disable some modules for example the dropout module
            self.encoder.eval()

        if self.use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            cands = cands.cuda()

        predictions, cand_preds, sparse_loss, grid_score, grid_mask = self.encoder(xs, cands)
        assert len(cand_preds.size()) == 2
        # sparse_cons = torch.mean(torch.cat([sparse_loss[ind][ys[ind]].view(-1)
        #                                     for ind in range(len(ys))]))
        sparse_cons = torch.mean(sparse_loss)
        # sparse_cons = sparse_loss
        loss = self.criterion(cand_preds, ys) + self.sparse * sparse_cons
        # save loss to metrics
        batch_size = cand_preds.size(0)
        num_cands = cand_preds.size(1)
        predicted_answer = cand_preds.max(1)[1]
        num_correct = predicted_answer.eq(ys).sum().item()
        # TEST CrossEntropy work as expectation
        # TODO add this part to the test module

        rank_cands = torch.argsort(cand_preds, dim=1, descending=True).data.cpu().numpy()
        for batch_ind, batch_answer in enumerate(all_answers):
            score = 0.0
            for answer in batch_answer:
                rank_ind = rank_cands[batch_ind].tolist().index(answer)
                score += 1.0 if rank_ind < len(batch_answer) else 0.0
            self.metrics['mean_avg_pre'] += score / len(batch_answer)

        self.metrics['num_correct'] += num_correct
        self.metrics['loss'] += loss.item()
        self.metrics['num_cands'] += num_cands
        self.metrics['num_exs'] += batch_size
        for i, y in enumerate(ys):
            self.metrics['positive_score_sum'] += cand_preds[i, y]

        self.metrics['score_sum'] += cand_preds.mean(dim=1).sum().item()
        self.metrics['score_std'] += cand_preds.std(dim=1).sum().item()

        if is_training:
            loss *= self.lr_decay[0]
            loss.backward()
            self.update_params()

        torch.cuda.empty_cache()

        return predictions, cand_preds, grid_score, grid_mask

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path
        if path is None:
            return

        model = {'model': self.encoder.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'optimizer_type': self.opt['optimizer']}

        with open(path, 'wb') as f:
            torch.save(model, f)

        self.opt['learningrate'] = self.lr * self.lr_decay[0]

        with open(path + ".opt", 'wb') as handle:
            pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if not os.path.isfile(path + '.opt'):
            # backwards compatible to old models
            self.opt = self.override_opt(states['opt'])
            # save .opt file to make compatible
            with open(path + ".opt", 'wb') as handle:
                pickle.dump(self.opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return states

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'sent_hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'attention',
                      'attention_length', 'rnn_class', 'dialog_embedding_size'}
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

    def shrink_lr(self):
        self.lr_decay[0] *= 0.1

    def report(self):
        m = {}
        num_cands = self.metrics['num_cands']
        if num_cands > 0:
            if self.metrics['num_correct'] > 0:
                m['token_acc'] = self.metrics['num_correct'] / num_cands
            m['loss'] = self.metrics['loss'] / num_cands
        num_exs = self.metrics.get('num_exs', 0)
        m['lr'] = self.opt['learningrate'] * self.lr_decay[0]
        if num_exs:
            m['pscore'] = self.metrics['positive_score_sum'] / num_exs
            m['score'] = self.metrics['score_sum'] / num_exs
            m['std'] = self.metrics['score_std'] / num_exs
            m['mean_avg_pre'] = self.metrics['mean_avg_pre'] / num_exs
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 12)
        return m


def reverse_padding(xs, PAD_IDX=0):
    """
    Move the PAD_IDX in front of the dialog

    :param xs: input dialogs, which are encoded by dictionary,
    :param PAD_IDX: the index of the __NULL__

    Examples
    --------
    >>> xs = [[3, 1, 2, 0, 0],
    ...       [2, 1, 4, 0, 0]]
    >>> reverse_padding(xs, 0)
    [[0, 0, 3, 1, 2],
     [0, 0, 2, 1, 4]]

    """
    if not isinstance(xs, list):
        xs = [[x for x in ex] for ex in xs]

    ans = []
    if len(xs) == 0:
        return xs
    n = len(xs[0])
    for line in xs:
        end_idx = n - 1
        for end_idx in range(n - 1, -1, -1):
            if line[end_idx] != PAD_IDX:
                break
        end_idx += 1
        padding_num = n - end_idx
        new_line = [PAD_IDX] * padding_num + line[:end_idx]
        ans.append(new_line)
    return ans


class ReceiverEncoder(nn.Module):
    """
    ReceiverEncoder

    opt, n_features, padding_idx=0, start_idx=1, end_idx=2, longest_label=1

    ----------------


    """
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, opt, n_features,
                 padding_idx=0, start_idx=1, end_idx=2, unk_idx=3, longest_label=1):
        super().__init__()
        self.opt = opt

        # self.attn_type = opt['attention']
        self.id = 'ReceiverRepeatLabelAgent'
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.START_IDX = start_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label
        self.bidirectional = opt['bidirectional']
        self.attn_type = opt['attention']

        bi = 1
        if opt['bidirectional']:
            bi = 2

        rnn_class = ReceiverEncoder.RNN_OPTS[opt['rnn_class']]
        self.sent_encoder = BertWrapper(bert_model=BertModel.from_pretrained(opt['bert_pretrained_path']),
                                        output_dim=opt['sent_hiddensize'],
                                        layer_pulled=-1)

        if opt['dialog_embedding_size'] == -1:
            embedding_size = 768
        else:
            embedding_size = opt['dialog_embedding_size']
        self.scaled_value = int(math.sqrt(embedding_size * bi))
        self.step = torch.zeros(1, dtype=torch.long, requires_grad=False)
        # bilinear W
        self.bilinear_weight = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.bilinear_weight.data.uniform_(-0.1, 0.1)

    def _encode_sent(self, xs):
        """

        :param xs:
            This is supposed to be 3d data with size [num_batch, num_dialog, num_words].
            1. multiple agents
            2. multiple dialogs
            3. word ind
        :return: 4d data with size [num_batch, num_dialog, num_sent_embedding_size]
            1. multiple agents
            2. sent vectors

        """
        shape = xs.size()
        # mask : num_batch x num_utterance
        enc_mask = torch.sum(xs.ne(0).int(), dim=-1).ne(0)
        # num_batch x num_dialog x num_step
        xs = xs.view(shape[0] * shape[1], shape[2])
        mask = (xs != self.NULL_IDX).long()
        segment_ids = mask * 0
        enc_out = self.sent_encoder(xs, segment_ids, mask)
        # review the shape
        enc_out = enc_out.contiguous().view(shape[0], shape[1], -1)
        return enc_out, enc_mask

    def _encode_personae(self, personae):
        """
        :param personae: 4d data with size [num_batch, num_candidates, num_personae, num_words]
        :return: 4d data with size [num_batch, num_candidates, num_personae, persona_embedding_size]
        """
        shape = personae.size()
        personae = personae.view(shape[0] * shape[1] * shape[2], -1)
        mask = (personae != self.NULL_IDX).long()
        # segment id 1
        segment_ids = mask
        enc_out = self.sent_encoder(personae, segment_ids, mask)
        # num_batch x num_persona x persona_embedding_size
        enc_out = enc_out.contiguous().view(shape[0], shape[1], shape[2], -1)
        output = enc_out
        return output

    def _encode_dialog(self, xs, xs_mask, reduce=False):
        """

        :param xs: 3d data with size [num_batch, num_dialog, num_steps]
            1. multiple agents
            2. dialog vectors
        :param reduce: if reduce, mean the all vector into one vector
        :return: 2d data with size [num_batch, num_dialog_embedding_size]
            1. multiple agents dialog feature vectors
        """
        encode_output, hidden = self.dialog_encoder(xs)
        # TODO: not elegant, please use the idf to multiply
        if reduce:
            # if reduce, we should find the xs_ends
            xs_mask = xs_mask.unsqueeze(-1).expand_as(encode_output)
            one_minus_mask = (1.0 - xs_mask).byte()
            # masked mean value
            replaced_vector = encode_output.masked_fill(one_minus_mask, 0.0)
            value_sum = torch.sum(replaced_vector, dim=1)
            value_count = torch.sum(xs_mask.float(), dim=1)
            encode_output = value_sum / value_count
        return encode_output

    def forward(self, xs, cands=None):
        """
        :param xs: multiple dialog vector data.
            This is supposed to be 3d data with size [num_batch, num_dialog, num_words].
            1. multiple agents
            2. multiple dialogs
            3. word ind
        :param cands:
        :return: dialogue_features 3d with size [num_batch, num_steps, num_dialog_embedding_size]
                 cands_scores 2d with size [num_batch, num_cand]
        """
        dialogue_features, dialog_ends = self._encode_sent(xs)
        # num_batch x num_dialogue x dialogue_embedding
        # dialogue_features = self._encode_dialog(dialog_input, dialog_ends)
        dialogue_shape = dialogue_features.size()
        if cands is not None:
            # num_batch x num_cand x num_persona x persona_embedding
            candidate_persona_features = self._encode_personae(cands)
            persona_shape = candidate_persona_features.size()
            # num_candidates x num_persona x persona_embedding
            candidate_persona_features = candidate_persona_features.view(persona_shape[0] * persona_shape[1],
                                                                         persona_shape[2],
                                                                         -1)
            # expand the dialogue feature
            dialogue_features = dialogue_features.unsqueeze(1).expand(persona_shape[0],
                                                                      persona_shape[1],
                                                                      dialogue_shape[1],
                                                                      dialogue_shape[2])
            # num_candidates x num_dialogue x dialogue_embedding
            dialogue_features = dialogue_features.contiguous(). \
                view(persona_shape[0] * persona_shape[1], dialogue_shape[1], -1)

            if self.opt['score_method'] == 'bilinear':
                hidden_size = self.bilinear_weight.size(0)
                expand_weight = self.bilinear_weight.expand(persona_shape[0] * persona_shape[1],
                                                            hidden_size,
                                                            hidden_size)
                # intermediate caluclation
                intermediate = torch.matmul(dialogue_features, expand_weight)
                # multiply persona features
                feature_map = torch.matmul(intermediate, candidate_persona_features.transpose(1, 2))
            elif self.opt['score_method'] == 'dot':
                feature_map = torch.matmul(dialogue_features, candidate_persona_features.transpose(1, 2))
            elif self.opt['score_method'] == 'cos':
                # 2-norm matrix multiply
                norm_res = torch.matmul(dialogue_features.norm(p=2, dim=2, keepdim=True),
                                        candidate_persona_features.norm(p=2, dim=2, keepdim=True).transpose(1, 2))
                feature_map = torch.matmul(dialogue_features, candidate_persona_features.transpose(1, 2)) / norm_res
            else:
                raise NotImplementedError("Not support for score_method mode: {}".format(self.opt['score_method']))

            if self.opt['score_normalize'] == 'scale':
                feature_map = feature_map / self.scaled_value
            elif self.opt['score_normalize'] == 'sigmoid':
                feature_map = F.sigmoid(feature_map)
            elif self.opt['score_normalize'] == 'scalesigmoid':
                feature_map = feature_map / self.scaled_value
                feature_map = F.sigmoid(feature_map)
            elif self.opt['score_normalize'] == 'none':
                pass
            else:
                raise NotImplementedError(
                    "Not support for score_normalize mode: {}".format(self.opt['score_normalize']))

            # construct mask
            dialogue_mask = ((xs != self.NULL_IDX).sum(-1) != 0).long()
            # num_batch x num_cand x num_dialogue x 1
            dialogue_mask = dialogue_mask.unsqueeze(1).expand(persona_shape[0],
                                                              persona_shape[1],
                                                              dialogue_shape[1])
            # num_candidates x num_dialogue x 1
            dialogue_mask = (dialogue_mask.contiguous()) \
                .view(persona_shape[0] * persona_shape[1], dialogue_shape[1], 1)
            # num_candidates x 1 x num_persona
            persona_mask = ((cands != self.NULL_IDX).sum(-1) != 0).long(). \
                view(persona_shape[0] * persona_shape[1], 1, persona_shape[2])
            # num_candidates x num_dialogue x num_persona
            whole_mask = (dialogue_mask * persona_mask)
            grid_mask = whole_mask.view(persona_shape[0], persona_shape[1], dialogue_shape[1], persona_shape[2]) \
                .detach().cpu().numpy()
            # whole_mask = whole_mask.view(-1, dialogue_shape[1] * persona_shape[2])
            one_minus_mask = (1.0 - whole_mask).byte()

            # masked mean value
            replaced_vector = feature_map.masked_fill(one_minus_mask, 0.0)
            grid_score = replaced_vector.view(persona_shape[0], persona_shape[1],
                                              dialogue_shape[1], persona_shape[2]).detach().cpu().numpy()

            # add a dropout
            # ones = replaced_vector.data.new_ones(replaced_vector.shape[-1])
            # dropout_mask = torch.nn.functional.dropout(ones, 0.5, self.training, inplace=False)
            # replaced_vector = dropout_mask.unsqueeze(dim=0) * replaced_vector
            # replaced_vector = F.dropout(replaced_vector, p=0.2, training=self.training)

            if self.opt['score_train'] == 'sum':
                value_sum = replaced_vector.sum(dim=2).sum(dim=1)
                # # average on dialogue dimension, sum on persona dimension
                persona_count = whole_mask.sum(dim=2)
                value_count = torch.ne(persona_count,
                                       torch.zeros(persona_count.size(), device=whole_mask.device).long()).sum(
                    dim=1).float()
                # operations on the final score
                score = value_sum / value_count
                score = score.view(persona_shape[0], persona_shape[1])

                sparse_loss = torch.sum(torch.abs(replaced_vector), dim=1).sum(dim=1)
                sparse_loss = sparse_loss / value_count
                sparse_loss = sparse_loss.view(persona_shape[0], persona_shape[1])
            elif self.opt['score_train'] == 'soft':
                self.step = self.step + dialogue_shape[0]
                temperature = min((int(self.step / 5000) + 1) * 0.1, 1.5)
                # temperature for multiply, increasing from 0.1 to 1.5
                soft_logits = replaced_vector * temperature
                attention = F.softmax(soft_logits, dim=2)
                # element-wise multiplication
                value_sum = (attention * replaced_vector).sum(dim=2).sum(dim=1)
                # # average on dialogue dimension, sum on persona dimension
                persona_count = whole_mask.sum(dim=2)
                value_count = torch.ne(persona_count,
                                       torch.zeros(persona_count.size(), device=whole_mask.device).long()).sum(
                    dim=1).float()
                # operations on the final score
                score = value_sum
                score = score.view(persona_shape[0], persona_shape[1])

                sparse_loss = torch.sum(torch.abs(replaced_vector), dim=1).sum(dim=1)
                sparse_loss = sparse_loss / value_count
                sparse_loss = sparse_loss.view(persona_shape[0], persona_shape[1])

            elif self.opt['score_train'] == 'inc':
                self.step = self.step + dialogue_shape[0]
                p_val = int(self.step / 10000) + 1
                if p_val > 10:
                    value_sum = torch.max(replaced_vector, dim=2)[0]
                else:
                    value_sum = torch.norm(replaced_vector, p=p_val, dim=2)

                value_sum = value_sum.sum(dim=1)

                persona_count = whole_mask.sum(dim=2)
                value_count = torch.ne(persona_count,
                                       torch.zeros(persona_count.size(), device=whole_mask.device).long()).sum(
                    dim=1).float()

                score = value_sum / value_count
                score = score.view(persona_shape[0], persona_shape[1])

                # sparse loss should be applied into all cases
                sparse_loss = torch.sum(torch.abs(replaced_vector), dim=2).sum(dim=1)
                sparse_loss = sparse_loss / value_count
                sparse_loss = sparse_loss.view(persona_shape[0], persona_shape[1])
            else:
                raise NotImplementedError("Not support for score_strategy mode: {}".format(self.opt['score_train']))

            # Fix: there should be a dimension reduction, see https://arxiv.org/abs/1706.03762
            # score /= self.scaled_value
            return dialogue_features, score, sparse_loss, grid_score, grid_mask
        return dialogue_features, None, None, None, None

    def measure_persona_similarity(self, dialogue_features, persona_features):
        persona_shape = persona_features.size()
        if self.opt['score_method'] == 'bilinear':
            hidden_size = self.bilinear_weight.size(0)
            expand_weight = self.bilinear_weight.expand(persona_shape[0],
                                                        hidden_size,
                                                        hidden_size)
            # intermediate caluclation
            intermediate = torch.matmul(dialogue_features, expand_weight)
            # multiply persona features
            feature_map = torch.matmul(intermediate, persona_features.transpose(1, 2))
        elif self.opt['score_method'] == 'dot':
            feature_map = torch.matmul(dialogue_features, persona_features.transpose(1, 2))
        elif self.opt['score_method'] == 'cos':
            # 2-norm matrix multiply
            norm_res = torch.matmul(dialogue_features.norm(p=2, dim=2, keepdim=True),
                                    persona_features.norm(p=2, dim=2, keepdim=True).transpose(1, 2))
            feature_map = torch.matmul(dialogue_features, persona_features.transpose(1, 2)) / norm_res
        else:
            raise NotImplementedError("Not support for score_method mode: {}".format(self.opt['score_method']))

        if self.opt['score_normalize'] == 'scale':
            feature_map = feature_map / self.scaled_value
        elif self.opt['score_normalize'] == 'sigmoid':
            feature_map = F.sigmoid(feature_map)
        elif self.opt['score_normalize'] == 'scalesigmoid':
            feature_map = feature_map / self.scaled_value
            feature_map = F.sigmoid(feature_map)
        elif self.opt['score_normalize'] == 'none':
            pass
        else:
            raise NotImplementedError("Not support for score_normalize mode: {}".format(self.opt['score_normalize']))

        if self.opt['score_train'] == 'sum':
            score = feature_map.sum(dim=2)
        elif self.opt['score_train'] == 'inc' or self.opt['score_train'] == 'soft':
            score = torch.max(feature_map, dim=2)[0]
        else:
            raise NotImplementedError("Now support score_train mode")
        # score += torch.std(feature_map, dim=2)
        # normalize by sentence length
        score = score.data.cpu().numpy()
        # set up a upper bound and lower bound
        return score

    def encode_persona(self, personae, dict_agent, use_cuda=False):
        """

        :param personae, a list of string that need converting
        :param dict_agent, a DictionaryAgent object, used to txt2vec
        :param use_cuda, cuda or cpu
        :return: tensor with size [num_persona, num_persona_embedding_size]
        """
        curr_cands = [{'text': c} for c in personae]
        cuda_device = next(self.sent_encoder.parameters()).device
        cs, *_ = PaddingUtils.pad_text(curr_cands, dict_agent, null_idx=dict_agent[dict_agent.null_token], dq=False)
        cs = split_pad_vector(cs, self.END_IDX, self.NULL_IDX)
        cs = [[x for x in ex] for ex in cs]
        cs = torch.LongTensor(cs)
        if use_cuda:
            cs = cs.cuda(cuda_device)
        cs = cs.unsqueeze(0)
        # return all persona embedding
        return self._encode_personae(cs).squeeze(0).squeeze(0)

    def encode_dialog(self, dialogs, valid_mask, dict_agent, use_cuda=False):
        """

        :param dialogs: a list of dialog that need converting

            it should follow the format as:

            ```
            a0 __START__ b0 __END__ a1 __START__ b1 __END___ ...

            ```
        :param dict_agent: a DictionaryAgent obj, used to txt2vec
        :param use_cuda: cuda or cpu
        Note: The part between __START__ and ___END__ corresponds to texts which describes
              the target persona
        :return: embeded dialogs with size [num_dialogs, num_embedding_size]
        """
        # self.eval()
        _end_idx = dict_agent[dict_agent.end_token]
        _pad_idx = dict_agent[dict_agent.null_token]
        obs = [{'text': c} for c in dialogs]
        cuda_device = next(self.sent_encoder.parameters()).device
        xs, _, _, sort_ind, *_ = PaddingUtils.pad_text(obs, dict_agent,
                                                       null_idx=_pad_idx,
                                                       dq=False, eval_labels=True)
        xs = split_pad_vector(xs, _end_idx, _pad_idx)
        xs = torch.LongTensor(xs)
        if use_cuda:
            xs = xs.cuda(cuda_device)
        sorted_dialog_input, dialog_ends = self._encode_sent(xs)

        # recover dialog_input
        # sort_ind[0] = x means origin[x] = cur[0] ; in order to resort origin to origin, desort ind
        desorted_ind = np.array(sort_ind).argsort()
        dialog_input = sorted_dialog_input[desorted_ind]
        # return all dialogue embedding
        # dialog_output = self._encode_dialog(dialog_feature, dialog_mask).squeeze(0)
        # record indices
        valid_ind = [ind for ind, value in enumerate(valid_mask) if value[0] == 1]
        # valid_ind = [ind for ind, value in enumerate(receive_mask)]
        valid_ind = torch.tensor(valid_ind, device=dialog_input.device).long()
        # only take the ones whose mask is 1
        dialog_receive = torch.index_select(dialog_input, 1, valid_ind)
        return dialog_input, dialog_receive


class AttentionLayer(nn.Module):
    ACTIVES = {'none': lambda: (lambda x: x), 'tanh': nn.Tanh, 'relu': nn.ReLU}

    def __init__(self, key_size, activator='tanh', attn_type='general'):
        super().__init__()
        self.attn_type = attn_type
        self._attention_std = 0
        if attn_type == 'none':
            return
        self.hidden_size = key_size
        self.weight = nn.Parameter(torch.Tensor(key_size, key_size))
        self.bias = nn.Parameter(torch.Tensor(key_size))
        self.proj = nn.Parameter(torch.Tensor(key_size, 1))
        self.softmax = nn.Softmax(dim=0)
        self.weight.data.uniform_(-0.1, 0.1)
        self.proj.data.uniform_(-0.1, 0.1)
        self.bias.data.zero_()
        self.active = self.ACTIVES[activator]()
        self.scaled_value = int(math.sqrt(self.hidden_size))

    def forward(self, xes, hidden_states=None, mask=None, batch_first=True):
        """

        :param xes: if batch_first, vector with size of [batch_size, num_steps, embedding_size]
        :param hidden_states: vector with size of [batch_size, num_steps, hidden_size]
        :param batch_first:
        :param mask: avoid the null idx to summarize the attention
        :return: [batch_size, hidden_size]
        """
        if self.attn_type == 'none':
            return xes
        if batch_first:
            xes = xes.transpose(1, 0)
            if hidden_states is not None:
                hidden_states = hidden_states.transpose(1, 0)

        hs = self.hidden_size
        w = self.weight.unsqueeze(0).expand(xes.size(0), hs, hs)
        b = self.bias.unsqueeze(0).unsqueeze(0).expand(xes.size(0), xes.size(1), hs)
        p = self.proj.unsqueeze(0).expand(xes.size(0), hs, 1)
        squish = torch.bmm(xes, w) + b
        # scale
        squish = self.active(squish)
        attn = torch.bmm(squish, p).squeeze(dim=2)
        attn = attn / self.scaled_value
        if mask is not None:
            # other x num_batch
            mask = mask.transpose(0, 1)
            attn[~mask] = -1000000.0
        attn_norm = self.softmax(attn)
        if hidden_states is not None:
            ret = hidden_states * attn_norm.unsqueeze(2).expand_as(hidden_states)
        else:
            ret = xes * attn_norm.unsqueeze(2).expand_as(xes)
        ret = ret.sum(0)
        # BUG FIX: why transpose it again?
        # if batch_first:
        #     ret = ret.transpose(1, 0)
        self._attention_std = np.std(attn_norm.cpu().detach().numpy())
        return ret

    def attention_std(self):
        return self._attention_std


class BertWrapper(torch.nn.Module):
    """ Adds a optional transformer layer and a linear layer on top of BERT.
    """

    def __init__(self, bert_model, output_dim, layer_pulled=-1):
        super(BertWrapper, self).__init__()
        self.layer_pulled = layer_pulled
        # deduce bert output dim from the size of embeddings
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.add_transformer_layer = False
        self.output_dim = output_dim
        if output_dim != -1:
            # -1 means keep original
            self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)

        self.bert_model = bert_model
        self.bert_model.embeddings.word_embeddings.weight.requires_grad = False
        self.bert_model.embeddings.token_type_embeddings.weight.requires_grad = False

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask)
        # output_bert is a list of 12 (for bert base) layers.
        layer_of_interest = output_bert[self.layer_pulled]
        if self.add_transformer_layer:
            # Follow up by yet another transformer layer
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            embeddings = self.additional_transformer_layer(
                layer_of_interest, extended_attention_mask)
        else:
            embeddings = layer_of_interest

        embeddings = embeddings.mean(dim=1)
        # embeddings = embeddings.max(dim=1)[0]

        if self.output_dim != -1:
            embeddings = self.additional_linear_layer(embeddings)
        return embeddings
