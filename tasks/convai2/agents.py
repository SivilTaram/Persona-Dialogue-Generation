import os
import copy

from parlai.core.teachers import FbDialogTeacher
from parlai.tasks.convai2.build import build


def _path(opt, persona, use_cands=None, self_play=False):
    """

    :param opt:
    :param persona: string, e.g. 'self', 'other', 'both', 'all',
                                 'self_revised', 'other_revised',
                                 'both_revised', 'all_revised'
    :param use_cands:
    :param self_play:
    :return:
    """
    # Build the data if it doesn't exist.
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        print("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    cands = '' if use_cands else '_no_cands'
    if self_play:
        return os.path.join(opt['datapath'], 'ConvAI2', dt + '_selfplay' + '.txt')
    else:
        return os.path.join(opt['datapath'], 'ConvAI2', dt + cands + '.txt')


class OriginalPersonaTeacher(FbDialogTeacher):
    """Teacher, generate a persona in each act"""
    def __init__(self, opt, shared=None):
        """

        :param opt:
        :param shared:
        """
        assert 'personapath' in opt, 'Please specify the path for the persona-only file'
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, persona='self_original', self_play=True)
        super().__init__(opt, shared)


class OriginalTeacher(FbDialogTeacher):
    """Teacher, generate a persona in each act"""
    def __init__(self, opt, shared=None):
        """

        :param opt:
        :param shared:
        """
        assert 'personapath' in opt, 'Please specify the path for the persona-only file'
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, persona='self_original', self_play=False)
        super().__init__(opt, shared)


class RevisedPersonaTeacher(FbDialogTeacher):
    """Teacher, generate a persona in each act"""
    def __init__(self, opt, shared=None):
        """

        :param opt:
        :param shared:
        """
        assert 'personapath' in opt, 'Please specify the path for the persona-only file'
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, persona='self_revised', self_play=True)
        super().__init__(opt, shared)


class RevisedTeacher(FbDialogTeacher):
    """Teacher, generate a persona in each act"""
    def __init__(self, opt, shared=None):
        """

        :param opt:
        :param shared:
        """
        assert 'personapath' in opt, 'Please specify the path for the persona-only file'
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, persona='self_revised', self_play=False)
        super().__init__(opt, shared)