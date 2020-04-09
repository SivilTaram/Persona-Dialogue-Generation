import torch
from torch.optim import Optimizer
from torch.optim import Adam
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
from parlai.core.utils import _ellipse


def get_bert_optimizer(models, bert_learning_rate, base_learning_rate, weight_decay):
    """ Optimizes the network with AdamWithDecay
    """
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    base_parameters = []
    base_parameters_names = []
    no_decay = ['bias', 'gamma', 'beta']

    for model in models:
        for n, p in model.named_parameters():
            if p.requires_grad:
                # fine-tune BERT
                if any(t in n for t in ["bert_model", "bert"]):
                    if any(t in n for t in no_decay):
                        parameters_without_decay.append(p)
                        parameters_without_decay_names.append(n)
                    else:
                        parameters_with_decay.append(p)
                        parameters_with_decay_names.append(n)
                else:
                    base_parameters.append(p)
                    base_parameters_names.append(n)

    print('The following parameters will be optimized WITH decay:')
    print(_ellipse(parameters_with_decay_names, 5, ' , '))
    print('The following parameters will be optimized WITHOUT decay:')
    print(_ellipse(parameters_without_decay_names, 5, ' , '))
    print('The following parameters will be optimized NORMALLY:')
    print(_ellipse(base_parameters_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': weight_decay, 'lr': bert_learning_rate},
        {'params': parameters_without_decay, 'weight_decay': 0.0, 'lr': bert_learning_rate},
        {'params': base_parameters, 'weight_decay': weight_decay, 'lr': base_learning_rate}
    ]
    optimizer = AdamWithDecay(optimizer_grouped_parameters,
                              lr=base_learning_rate)
    return optimizer


class AdamWithDecay(Optimizer):
    """ Same implementation as Hugging's Face, since it seems to handle better the
        weight decay than the Pytorch default one.
        Stripped out of the scheduling, since this is done at a higher level
        in ParlAI.
    Params:
        lr: learning rate
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping).
                       Default: 1.0
    """

    def __init__(self, params, lr=required,
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                'Invalid b1 parameter: {} - should be in [0.0, 1.0['.format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                'Invalid b2 parameter: {} - should be in [0.0, 1.0['.format(b2))
        if not e >= 0.0:
            raise ValueError('Invalid epsilon value: {} - should be >= 0.0'.format(e))
        defaults = dict(lr=lr,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(AdamWithDecay, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please '
                        'consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data
                lr = group['lr']

                update_with_lr = lr * update
                p.data.add_(-update_with_lr)
        return loss
