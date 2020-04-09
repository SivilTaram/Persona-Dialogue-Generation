import os
import copy

from parlai.core.teachers import FbDialogTeacher
from agents.common.bert_dictonary import BertDictionaryAgent
from agents.common.gpt_dictionary import GPTDictionaryAgent
from parlai.core.dict import DictionaryAgent
from parlai.tasks.convai2.build import build
import random
import sys
import re
from agents.common.dict_helper import SpecialToken
sys.path.append('../.../../')
PERSONAE_NUM = 4


class ReceiverTeacher(FbDialogTeacher):
    """
    Do not use this class directly.
    Use teachers as BothOriginalTeacher, which set the opt['datafile']
    """
    DEFAULT_NEG_NUM = 1  # default negtive sampling num
    # DEFAULT_SAMPLE_METHOD = 'combination'
    DEFAULT_SAMPLE_METHOD = 'combination'

    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.sample_num = opt.get('negative_sample_num', self.DEFAULT_NEG_NUM)
        # {combination, origin}
        self.sample_method = opt.get('sample_method', self.DEFAULT_SAMPLE_METHOD)
        assert opt.get('datafile', False), "You should set opt['datafile'] before " \
                                           "using ReceiverTeacher"
        if shared is None:
            self.personae = self.load_all_persona(load_path=opt['persona_path'])
            print("[ReceiverTeacher uses sample method of {}]".format(self.sample_method))
            print("[ ReceiverTeacher negative sample num = {} personae num = {}]"
                  .format(self.sample_num, len(self.personae)))
        else:
            self.personae = shared['personae']

        # set default persona
        self.default_persona = SpecialToken.no_fact
        self.is_training = opt['datatype'].split(':')[0] == 'train'
        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['personae'] = self.personae
        return shared

    def load_all_persona(self, max_persona=100000, load_path='all'):
        def yield_persona(path):
            with open(path) as f:
                for line in f:
                    sep = line.find(' ')
                    line = line[sep + 1:-1]
                    yield line

        personae_data_path = _path(self.opt, load_path, True)
        # personae_data_path = _path(self.opt, '', True)
        personae = []
        for persona in yield_persona(personae_data_path):
            personae.append(persona)
            if len(personae) > max_persona:
                break
        if self.sample_method == 'combination':
            ret = []
            for persona in personae:
                # BUG FIX: split with `.` and then add it to the tail of utterance will cause the
                # inconsistent with the annotation, which removes the space before dot.
                # xs = [x.strip() for x in persona.split('.')]
                xs = [x.strip() for x in re.findall("[^\.]+\.", persona)]
                ret.extend([x for x in xs if len(x)])
            personae = ret
        return personae

    def _split_dialog(self, path):
        """

        :param path:
        :return: A list of dialog. Each element contains:

        {
            'dialog': [
                A_0,
                B_0,
                A_1,
                B_1,
                ...
            ],
            'persona_a': [], # a list of persona
            'persona_b': [], # a list of persona
        }
        """
        def _get_conv_id(_line):
            # first, get conversation index -- '1' means start of episode
            space_idx = _line.find(' ')
            if space_idx == -1:
                # empty line, both individuals are saying whitespace
                _conv_id = int(_line)
            else:
                _conv_id = int(_line[:space_idx])
            return _conv_id

        def which_persona(_line):
            if "partner's persona:" in _line:
                return 'persona_a'
            elif "your persona:" in _line:
                return 'persona_b'
            return None

        print("[loading fbdialog data:" + path + "]")
        ans = []
        with open(path) as read:
            x = {'dialog': [], 'persona_a': [], 'persona_b': []}
            last_conv_id = None
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    # empty response
                    continue

                conv_id = _get_conv_id(line)
                if last_conv_id is None or conv_id <= last_conv_id:
                    if len(x['dialog']):
                        ans.append(x)
                    x = {'dialog': [], 'persona_a': [], 'persona_b': []}
                last_conv_id = conv_id

                space_idx = line.find(' ')
                split = line[space_idx + 1:].split('\t')
                which = which_persona(line)
                if which is None:
                    # if dialog
                    dialog_a, dialog_b = split[0], split[1]
                    x['dialog'].append(dialog_a)
                    x['dialog'].append(dialog_b)
                else:
                    # if persona
                    persona = split[0].split(':', maxsplit=2)[1].strip()
                    x[which].append(persona)
        return ans

    def create_persona(self, existing_persona):
        """
        create persona which owes the same length
        :param existing_persona:
        :return:
        """
        # here the __non__fact__
        persona = existing_persona.copy()
        # add_default = self.default_persona in persona
        # random mask 2
        replace_indices = random.sample(range(len(persona)), len(persona))
        replaced_number = len(replace_indices)
        append_persona = random.sample(self.personae, replaced_number)
        while bool(set(append_persona) & set(persona)):
            append_persona = random.sample(self.personae, replaced_number)
        for ind in replace_indices:
            # if add default, it is the last one
            # if add_default and ind == len(persona) - 1:
            #     continue
            persona[ind] = append_persona.pop()
        # add default personae
        # persona.append(self.default_persona)
        return persona

    def sample_candidates(self, existing_persona):
        if self.sample_method == 'origin':
            return random.sample(self.personae, self.sample_num)
        elif self.sample_method == 'combination':
            # random select three of them, and replace them with sample one
            personae = []
            # every number should sample three times
            for _ in range(self.sample_num):
                personae.append(self.create_persona(existing_persona))
            return personae
        else:
            raise NotImplemented()

    def setup_data(self, path):
        def yield_dialog(_dialog, _is_training):
            speaker = 0
            x = ''
            for d in _dialog['dialog']:
                d = d.replace('\n', ' ')
                # if speaker == 1:
                #     x += start_token + d + end_token
                if speaker == 0:
                    if d != '':
                        x += start_token + d + end_token + ' '
                # elif speaker == 1:
                #     x += ' ' + start_token + d + end_token
                speaker = 1 - speaker
            # speaker 1 means person b
            # random sampling add default persona, 0.05 probability
            add_default_persona = random.random() > 1.01
            if add_default_persona and _is_training:
                dialog_persona = _dialog['persona_a'] + [self.default_persona]
            else:
                dialog_persona = _dialog['persona_a']
            candidates = [dialog_persona] + self.sample_candidates(dialog_persona)
            candidates_split = []
            for candidate_persona in candidates:
                persona_repr = ''
                for _persona in candidate_persona:
                    persona_repr += start_token + _persona + end_token + ' '
                persona_repr = persona_repr.strip()
                candidates_split.append(persona_repr)

            return [x, [candidates_split[0]], 1, candidates_split], True

        def reverse_dialog(_dialog):
            _dialog = copy.deepcopy(_dialog)
            _dialog['dialog'] = [''] + _dialog['dialog']
            _dialog['persona_a'], _dialog['persona_b'] = _dialog['persona_b'], _dialog['persona_a']
            return _dialog

        is_training = 'train' in path
        dialogs = self._split_dialog(path)

        # WARNING: use default to split dialogues
        end_token = ' {} '.format(BertDictionaryAgent.default_end)
        start_token = ' {} '.format(BertDictionaryAgent.default_start)

        for dialog in dialogs:
            if dialog['persona_a']:
                yield yield_dialog(dialog, is_training)
            if dialog['persona_b']:
                dialog = reverse_dialog(dialog)
                yield yield_dialog(dialog, is_training)


def _path(opt, persona, no_cands):
    # Build the data if it doesn't exist.
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        print("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    # dialogue data
    dt = datatype + '_' + persona
    cands = '' if no_cands else '_no_cands'
    return os.path.join(opt['datapath'], 'ConvAI2', dt + cands + '.txt')


class BothOriginalTeacher(ReceiverTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_original', use_cands)
        opt['persona_path'] = 'receiver_both_original'
        super().__init__(opt, shared)


class BothRevisedTeacher(ReceiverTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_revised', use_cands)
        opt['persona_path'] = 'receiver_both_revised'
        super().__init__(opt, shared)


class BothTeacher(BothOriginalTeacher):
    pass


class DefaultTeacher(BothOriginalTeacher):
    pass


class ReceiverRankingTeacher(FbDialogTeacher):
    """
    Do not use this class directly.
    Use teachers as BothOriginalTeacher, which set the opt['datafile']
    """

    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.personae = list(set(self.load_all_persona()))

        # set default persona
        self.is_training = opt['datatype'].split(':')[0] == 'train'
        super().__init__(opt, shared)

    def share(self):
        shared = super().share()
        shared['personae'] = self.personae
        return shared

    def load_all_persona(self):
        load_path = self.opt['cands_path']
        personae_data_path = _path(self.opt, load_path, True)
        with open(personae_data_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            personae = [line.strip() for line in lines]
        return personae

    def _split_dialog(self, path):
        """

        :param path:
        :return: A list of dialog. Each element contains:

        {
            'dialog': [
                A_0,
                B_0,
                A_1,
                B_1,
                ...
            ],
            'persona_a': [], # a list of persona
            'persona_b': [], # a list of persona
        }
        """
        def _get_conv_id(_line):
            # first, get conversation index -- '1' means start of episode
            space_idx = _line.find(' ')
            if space_idx == -1:
                # empty line, both individuals are saying whitespace
                _conv_id = int(_line)
            else:
                _conv_id = int(_line[:space_idx])
            return _conv_id

        def which_persona(_line):
            if "partner's persona:" in _line:
                return 'persona_a'
            elif "your persona:" in _line:
                return 'persona_b'
            return None

        print("[loading fbdialog data:" + path + "]")
        ans = []
        with open(path) as read:
            x = {'dialog': [], 'persona_a': [], 'persona_b': []}
            last_conv_id = None
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    # empty response
                    continue

                conv_id = _get_conv_id(line)
                if last_conv_id is None or conv_id <= last_conv_id:
                    if len(x['dialog']):
                        ans.append(x)
                    x = {'dialog': [], 'persona_a': [], 'persona_b': []}
                last_conv_id = conv_id

                space_idx = line.find(' ')
                split = line[space_idx + 1:].split('\t')
                which = which_persona(line)
                if which is None:
                    # if dialog
                    dialog_a, dialog_b = split[0], split[1]
                    x['dialog'].append(dialog_a)
                    x['dialog'].append(dialog_b)
                else:
                    # if persona
                    persona = split[0].split(':', maxsplit=2)[1].strip()
                    x[which].append(persona)
        return ans

    def setup_data(self, path):
        def yield_dialog(_dialog):
            speaker = 0
            x = ''
            for d in _dialog['dialog']:
                d = d.replace('\n', ' ')
                # if speaker == 1:
                #     x += start_token + d + end_token
                if speaker == 0:
                    if d != '':
                        x += start_token + d + end_token + ' '
                # elif speaker == 1:
                #     x += ' ' + start_token + d + end_token
                speaker = 1 - speaker
            # speaker 1 means person b
            # random sampling add default persona, 0.05 probability
            dialog_persona = _dialog['persona_a']
            return [x, dialog_persona, 1, self.personae], True

        def reverse_dialog(_dialog):
            _dialog = copy.deepcopy(_dialog)
            _dialog['dialog'] = [''] + _dialog['dialog']
            _dialog['persona_a'], _dialog['persona_b'] = _dialog['persona_b'], _dialog['persona_a']
            return _dialog

        dialogs = self._split_dialog(path)

        # WARNING: use default to split dialogues
        end_token = ' {} '.format(BertDictionaryAgent.default_end)
        start_token = ' {} '.format(BertDictionaryAgent.default_start)
        for dialog in dialogs:
            if dialog['persona_a']:
                yield yield_dialog(dialog)
            if dialog['persona_b']:
                dialog = reverse_dialog(dialog)
                yield yield_dialog(dialog)


class BothOriginalRankingTeacher(ReceiverRankingTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_original', use_cands)
        opt['cands_path'] = 'receiver_original_candspair'
        super().__init__(opt, shared)


class BothRevisedRankingTeacher(ReceiverRankingTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_revised', use_cands)
        opt['cands_path'] = 'receiver_revised_candspair'
        super().__init__(opt, shared)
