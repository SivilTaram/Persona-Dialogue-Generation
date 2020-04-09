from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
from parlai.core.logs import TensorboardLogger
from parlai.scripts.build_dict import build_dict, setup_args as setup_dict_args
import random
import torch
import math
import os
from agents.common.dict_helper import SpecialToken

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# if is original, train model on original data; otherwise on revised data.
IS_ORIGINAL = False

RECEIVER_DIR = './tmp/receiver'
VERSION = 'receiver_revised'


def setup_task():
    if IS_ORIGINAL:
        task_name = 'tasks.convai2receiver.agents:BothOriginalTeacher'
    else:
        task_name = 'tasks.convai2receiver.agents:BothRevisedTeacher'
    return task_name


def setup_seed(seed=1706123):
    # random seed, to evaluate the performance
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def setup_args():
    """
    Use create test env setting
    :return: opt
    """
    parser = setup_dict_args()

    # 'dot', 'bilinear', 'cos'
    score_method = 'dot'
    # 'scale', 'sigmoid', 'none', 'scalesigmoid'
    score_normalize = 'scale'
    # 'sum', 'inc', 'soft'
    score_train = 'soft'

    batch_size = 2
    validation_batch_size = 8
    candidates_num = 2
    val_candidates_num = 32
    sent_hidden_size = -1
    dialog_hidden_size = -1
    weight_decay = 1e-5
    learning_rate = 1e-4
    bert_learning_rate = 6.25e-6
    dropout = 0.1
    input_dropout = 0
    bidirectional = False
    sparse = 1e-2
    # {'none', 'general'}
    attention = 'general'
    # {'cross_entropy', 'margin'}
    criterion = 'margin'
    marginloss_margin = 0.5

    optim_method = 'adam'
    validation_every_n_secs = 2400
    log_every_n_secs = 20
    num_epoch = 50
    validation_exs = 5000
    lr_impatience = 5
    numlayers = 1

    cri = criterion
    if criterion == 'margin':
        cri += str(marginloss_margin)

    exp_name = VERSION

    task_name = setup_task()
    parser.set_defaults(
        exp=exp_name,  # name for experiment
        task=task_name,
        batchsize=batch_size,
        validation_batch_size=validation_batch_size,
        dict_include_valid=True,
        dict_tokenizer='split',
        dict_nulltoken=SpecialToken.pad,
        dict_starttoken=SpecialToken.start,
        dict_endtoken=SpecialToken.end,
        dict_unktoken=SpecialToken.unk,
        datatype='train',
        # model configuration
        model='agents.receiver.receiver:ReceiverAgent',
        model_file=os.path.join(os.path.abspath(RECEIVER_DIR), '{}.model'.format(exp_name)),
        init_transmitter=os.path.join(os.path.abspath(RECEIVER_DIR), '{}.model'.format(exp_name)),
        # validation configuration
        validation_every_n_secs=validation_every_n_secs,
        # in default, this can be chosen from hit@k, f1, bleu, accuracy
        # here only accuracy and hit@k make sense
        validation_metric='hits@1',
        validation_metric_mode='max',
        # Stop training when the metrics cannot meet the best result for
        # validation_patience times.
        validation_patience=15,
        log_every_n_secs=log_every_n_secs,
        # device configuration
        gpu=0,
        tensorboard_log=True,
        tensorboard_tag='exp',
        tensorboard_metrics='loss,hits@1,hits@5,hits@10,lr',
        # teacher negative sampling num
        negative_sample_num=candidates_num - 1,
        valid_negative_sample_num=val_candidates_num - 1,
        # can be set as all, which will show all
        # the metrics in training
        metrics='loss,hits@1,lr',
        tensorboard_comment='',  # add to the tensorboard output
        num_epochs=num_epoch,  # total number of epochs
        max_train_time=60000,
        save_every_n_secs=1200,
        validation_every_n_epochs=1,
        # enable this when debugging
        display_examples=False,
        # limitation of the number of examples if exs > 0
        validation_max_exs=validation_exs,
        # when the accuracy meet this value, stop training
        validation_cutoff=0.999,
        no_cuda=False,
        # {'random', 'glove', 'glove_fixed'}
        embedding_type="glove_fixed",
        embeddingsize=300,
        sent_hiddensize=sent_hidden_size,
        dialog_embedding_size=dialog_hidden_size,
        learningrate=learning_rate,
        weight_decay=weight_decay,
        bert_learning_rate=bert_learning_rate,
        bidirectional=bidirectional,
        dropout=dropout,
        input_dropout=input_dropout,
        attention=attention,
        criterion=criterion,
        lr_impatience=lr_impatience,
        marginloss_margin=marginloss_margin,
        optimizer=optim_method,
        numlayers=numlayers,
        sparse=sparse,
        score_train=score_train,
        score_normalize=score_normalize,
        score_method=score_method
    )
    opt = parser.parse_args([])
    # Override the setting which is saved in the .opt file
    opt['override'] = dict(
        # Whether load model from the latest checkpoint
        load_from_checkpoint=True
    )
    return opt


def run_eval(agent, opt, datatype, max_exs=-1, write_log=False, valid_world=None):
    """Eval on validation/test data.
    - Agent is the agent to use for the evaluation.
    - opt is the options that specific the task, eval_task, etc
    - datatype is the datatype to use, such as "valid" or "test"
    - write_log specifies to write metrics to file if the model_file is set
    - max_exs limits the number of examples if max_exs > 0
    - valid_world can be an existing world which will be reset instead of reinitialized
    """
    print('[ running eval: ' + datatype + ' ]')
    if 'stream' in opt['datatype']:
        datatype += ':stream'
    opt['datatype'] = datatype
    opt['negative_sample_num'] = opt['valid_negative_sample_num']
    if valid_world is None:
        # reset the validation batch size to accelerate validation
        opt['batch_size'] = opt['validation_batch_size']
        valid_world = create_task(opt, agent)
    valid_world.reset()
    cnt = 0
    while not valid_world.epoch_done():
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            print(valid_world.display() + '\n~~')
            print(valid_world.report())
        cnt += opt['batchsize']
        if max_exs > 0 and cnt > max_exs + opt.get('numthreads', 1):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_report = valid_world.report()
    valid_world.reset()  # this makes sure agent doesn't remember valid data

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    if write_log and opt.get('model_file'):
        # Write out metrics
        f = open(opt['model_file'] + '.' + datatype, 'a+')
        f.write(metrics + '\n')
        f.close()

    return valid_report, valid_world


def save_best_valid(model_file, best_valid):
    f = open(model_file + '.best_valid', 'w')
    f.write(str(best_valid))
    f.close()


class TrainLoop:
    def __init__(self, opt):
        if isinstance(opt, ParlaiParser):
            print('[ Deprecated Warning: TrainLoop should be passed opt not Parser ]')
            opt = opt.parse_args()
        # Possibly load from checkpoint
        if opt['load_from_checkpoint'] and opt.get('model_file') and os.path.isfile(opt['model_file'] + '.checkpoint'):
            opt['init_model'] = opt['model_file'] + '.checkpoint'
        # Possibly build a dictionary (not all models do this).
        if 'dict_file' not in opt or opt['dict_file'] is None:
            opt['dict_file'] = opt['model_file'] + '.dict'
        print("[ building dictionary first... ]")
        build_dict(opt, skip_if_built=True)
        # Create model and assign it to the specified task
        self.agent = create_agent(opt)
        self.world = create_task(opt, self.agent)
        self.train_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()
        print('[ training... ]')
        self.parleys = 0
        self.max_num_epochs = opt['num_epochs'] if opt['num_epochs'] > 0 else float('inf')
        self.max_train_time = opt['max_train_time'] if opt['max_train_time'] > 0 else float('inf')
        self.log_every_n_secs = opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 else float('inf')
        self.val_every_n_secs = opt['validation_every_n_secs'] if opt['validation_every_n_secs'] > 0 else float('inf')
        self.save_every_n_secs = opt['save_every_n_secs'] if opt['save_every_n_secs'] > 0 else float('inf')
        self.val_every_n_epochs = opt['validation_every_n_epochs'] if opt['validation_every_n_epochs'] > 0 else float(
            'inf')
        self.last_valid_epoch = 0
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.best_valid = None
        if opt.get('model_file') and os.path.isfile(opt['model_file'] + '.best_valid'):
            with open(opt['model_file'] + ".best_valid", 'r') as f:
                x = f.readline()
                self.best_valid = float(x)
                f.close()
        self.impatience = 0
        self.lr_impatience = 0
        self.saved = False
        self.valid_world = None
        self.opt = opt
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)

    def validate(self):
        opt = self.opt
        # run evaluation on valid set
        valid_report, self.valid_world = run_eval(
            self.agent, opt, 'valid', opt['validation_max_exs'],
            valid_world=self.valid_world)

        # logging
        if opt['tensorboard_log'] is True:
            self.writer.add_metrics('valid', self.world.get_total_exs(), valid_report)
        # saving
        if opt.get('model_file') and opt.get('save_after_valid'):
            print("[ saving model checkpoint: " + opt['model_file'] + ".checkpoint ]")
            self.agent.save(opt['model_file'] + '.checkpoint')

        # send valid metrics to agent if the agent wants them
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)

        # check which metric to look at
        if '/' in opt['validation_metric']:
            # if you are multitasking and want your validation metric to be
            # a metric specific to a subtask, specify your validation metric
            # as -vmt subtask/metric
            subtask = opt['validation_metric'].split('/')[0]
            validation_metric = opt['validation_metric'].split('/')[1]
            new_valid = valid_report['tasks'][subtask][validation_metric]
        else:
            new_valid = valid_report[opt['validation_metric']]

        # check if this is the best validation so far
        if self.best_valid is None or self.valid_optim * new_valid > self.valid_optim * self.best_valid:
            print('[ new best {}: {}{} ]'.format(
                opt['validation_metric'], new_valid,
                ' (previous best was {})'.format(self.best_valid) if self.best_valid is not None else ''))
            self.best_valid = new_valid
            self.impatience = 0
            self.lr_impatience = 0
            if opt.get('model_file'):
                print("[ saving best valid model: " + opt['model_file'] + " ]")
                self.agent.save(opt['model_file'])
                print("[ saving best valid metric: " + opt['model_file'] + ".best_valid ]")
                save_best_valid(opt['model_file'], self.best_valid)
                self.saved = True
            if opt['validation_metric'] == 'accuracy' and self.best_valid >= opt['validation_cutoff']:
                print('[ task solved! stopping. ]')
                return True
        else:
            self.impatience += 1
            self.lr_impatience += 1
            print('[ did not beat best {}: {} impatience: {} ]'.format(
                opt['validation_metric'], round(self.best_valid, 4),
                self.impatience))
            if self.lr_impatience == opt['lr_impatience']:
                self.lr_impatience = 0
                self.agent.shrink_lr()
        self.validate_time.reset()

        # check if we are out of patience
        if 0 < opt['validation_patience'] <= self.impatience:
            print('[ ran out of patience! stopping training. ]')
            return True
        return False

    def log(self):
        opt = self.opt
        if opt['display_examples']:
            print(self.world.display() + '\n~~')
        logs = []
        # get report
        train_report = self.world.report()
        self.world.reset_metrics()

        # time elapsed
        logs.append('time:{}s'.format(math.floor(self.train_time.time())))
        logs.append('total_exs:{}'.format(self.world.get_total_exs()))

        exs_per_ep = self.world.num_examples()
        if exs_per_ep:
            logs.append('total_eps:{}'.format(
                round(self.world.get_total_exs() / exs_per_ep, 2)))

        if 'time_left' in train_report:
            logs.append('time_left:{}s'.format(
                math.floor(train_report.pop('time_left', ""))))
        if 'num_epochs' in train_report:
            logs.append('num_epochs:{}'.format(
                train_report.pop('num_epochs', '')))
        log = '[ {} ] {}'.format(' '.join(logs), train_report)
        print(log)
        self.log_time.reset()

        if opt['tensorboard_log'] is True:
            self.writer.add_metrics('train', self.world.get_total_exs(), train_report)

    def train(self):
        opt = self.opt
        world = self.world
        with world:
            while True:
                # do one example / batch of examples
                world.parley()
                self.parleys += 1

                # check counters and timers
                if world.get_total_epochs() >= self.max_num_epochs:
                    self.log()
                    print('[ num_epochs completed:{} time elapsed:{}s ]'.format(
                        self.max_num_epochs, self.train_time.time()))
                    break
                if self.train_time.time() > self.max_train_time:
                    print('[ max_train_time elapsed:{}s ]'.format(self.train_time.time()))
                    break
                if self.log_time.time() > self.log_every_n_secs:
                    self.log()
                if self.validate_time.time() > self.val_every_n_secs:
                    stop_training = self.validate()
                    if stop_training:
                        break
                if world.get_total_epochs() - self.last_valid_epoch >= self.val_every_n_epochs:
                    stop_training = self.validate()
                    self.last_valid_epoch = world.get_total_epochs()
                    if stop_training:
                        break
                if self.save_time.time() > self.save_every_n_secs and opt.get('model_file'):
                    print("[ saving model checkpoint: " + opt['model_file'] + ".checkpoint ]")
                    self.agent.save(opt['model_file'] + '.checkpoint')
                    self.save_time.reset()

        if not self.saved:
            # save agent
            self.agent.save(opt['model_file'])
        elif opt.get('model_file'):
            # reload best validation model
            self.agent = create_agent(opt)

        v_report, v_world = run_eval(self.agent, opt, 'valid', write_log=True)
        t_report, t_world = run_eval(self.agent, opt, 'test', write_log=True)
        v_world.shutdown()
        t_world.shutdown()
        return v_report, t_report


if __name__ == '__main__':
    opt = setup_args()
    setup_seed()
    TrainLoop(opt).train()
