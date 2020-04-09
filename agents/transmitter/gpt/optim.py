from pytorch_pretrained_bert import OpenAIAdam
from parlai.core.utils import _ellipse


class GPTOptimizer:
    def __init__(self, model, opt):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_with_decay = []
        parameters_with_decay_names = []
        parameters_without_decay = []
        parameters_without_decay_names = []
        base_parameters = []
        base_parameters_names = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                # fine-tune BERT
                if any(t in n for t in ["transformer"]):
                    if any(t in n for t in no_decay):
                        parameters_without_decay.append(p)
                        parameters_without_decay_names.append(n)
                    else:
                        parameters_with_decay.append(p)
                        parameters_with_decay_names.append(n)
                else:
                    base_parameters.append(p)
                    base_parameters_names.append(n)

        weight_decay = opt['weight_decay']
        bert_learning_rate = opt['gpt_lr']
        base_learning_rate = opt['lr']
        optimizer_grouped_parameters = [
            {'params': parameters_with_decay, 'weight_decay': weight_decay, 'lr': bert_learning_rate},
            {'params': parameters_without_decay, 'weight_decay': 0.0, 'lr': bert_learning_rate},
            {'params': base_parameters, 'weight_decay': weight_decay, 'lr': base_learning_rate}
        ]
        #
        print('The following parameters will be optimized WITH decay:')
        print(_ellipse(parameters_with_decay_names, 5, ' , '))
        print('The following parameters will be optimized WITHOUT decay:')
        print(_ellipse(parameters_without_decay_names, 5, ' , '))
        print('The following parameters will be optimized NORMALLY:')
        print(_ellipse(base_parameters_names, 5, ' , '))

        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=opt['gpt_lr'],
                               warmup=opt['warmup_proportion'],
                               max_grad_norm=opt['gradient_clip'],
                               t_total=opt.get('optimizer_step', -1))
        self.optimizer = optimizer

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self.optimizer.step()
