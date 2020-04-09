"""Traing model via self-play
Training methods:
    - Recurrent Policy Gradient
    - Batch Policy Gradient
    - Dual Policy Gradient
"""
# TODO: confirm dual learning training method
from worlds.selfplay import create_selfplay_world
from scripts.build_dict import build_dict, setup_args as setup_dict_args

from parlai.core.worlds import create_task
from parlai.core.agents import create_agent, create_agent_from_shared
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
from parlai.core.logs import TensorboardLogger
import math
import os
from copy import deepcopy


def _setup_op(opt, component='transmitter'):
    assert component in ['transmitter', 'receiver']
    new_opt = deepcopy(opt)
    for k,v in opt.items():
        if k.endswith(component):
            new_k = k[:-len(component) - 1]
            new_opt[new_k] = v
    return new_opt


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True)
    train = parser.add_argument_group('Training Loop Arguments')
    train.add_argument('-dbf', '--dict-build-first',
                       type='bool', default=True,
                       help='build dictionary first before training agent')
    train.add_argument('-eps', '--num-epochs', type=float, default=-1)
    train.add_argument('-ttim', '--max-train-time',
                       type=float, default=-1)
    train.add_argument('-ltim', '--log-every-n-secs',
                       type=float, default=2)
    train.add_argument('-tden', '--train-display-every-n-secs',
                       type=float, default=-1,
                       help='Display training information every n seconds')
    train.add_argument('-vtim', '--validation-every-n-secs',
                       type=float, default=-1,
                       help='Validate every n seconds. Whenever the the best '
                            'validation metric is found, saves the model to '
                            'the model_file path if set.')
    train.add_argument('-stim', '--save-every-n-secs',
                       type=float, default=-1,
                       help='Saves the model to model_file.checkpoint after '
                            'every n seconds (default -1, never).')

    TensorboardLogger.add_cmdline_args(parser)
    parser = setup_dict_args(parser)
    return parser


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
    if opt.get('evaltask'):
        opt['task'] = opt['evaltask']
    if opt.get('evalbatchsize'):
        opt['batchsize'] = opt['evalbatchsize']

    if valid_world is None:
        valid_world = create_task(opt, agent)
    valid_world.reset()
    cnt = 0
    while not valid_world.epoch_done():
        if cnt % 1000 == 0:
            print('###### Parley {} ...'.format(cnt))
        valid_world.parley()
        if cnt == 0 and opt['display_examples']:
            print('display examples ... ...')
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
    if write_log and opt.get('model_file_transmitter') and opt.get('model_file_receiver'):
        # Write out metrics
        transmitter = opt['model_file_transmitter'].split('/')[-1]
        receiver = opt['model_file_receiver'].split('/')[-1]
        f = open(opt['model_file'] + transmitter + '_' + receiver + '.' + datatype, 'a+')
        f.write(metrics + '\n')
        f.close()

    return valid_report, valid_world


def save_best_valid(model_file, best_valid):
    f = open(model_file + '.best_valid', 'w')
    f.write(str(best_valid))
    f.close()


class TrainLoop():
    def __init__(self, opt):
        if isinstance(opt, ParlaiParser):
            opt = opt.parse_args()
        # Possibly build a dictionary (not all models do this).
        if opt['dict_build_first'] and 'dict_file' in opt:
            if opt['dict_file'] is None and opt.get('model_file_transmitter') and opt.get('model_file_receiver'):
                opt['dict_file'] = opt['model_file_transmitter'] + '_' + opt['model_file_receiver']  + '.dict'
            print("[ building dictionary first... ]")
            build_dict(opt, skip_if_built=False)

        # Create model and assign it to the specified task
        print("[ create meta-agent ... ]")
        self.agent = create_agent(opt)
        print("[ create agent A ... ]")
        shared = self.agent.share()
        self.agent_a = create_agent_from_shared(shared)
        self.agent_a.set_id(suffix=' A')
        print("[ create agent B ... ]")
        self.agent_b = create_agent_from_shared(shared)
        # self.agent_b = create_agent(opt)
        self.agent_b.set_id(suffix=' B')
        # self.agent_a.copy(self.agent, 'transmitter')
        # self.agent_b.copy(self.agent, 'transmitter')
        self.world = create_selfplay_world(opt, [self.agent_a, self.agent_b])

        # TODO: if batch, it is also not parallel
        # self.world = BatchSelfPlayWorld(opt, self_play_world)

        self.train_time = Timer()
        self.train_dis_time = Timer()
        self.validate_time = Timer()
        self.log_time = Timer()
        self.save_time = Timer()
        print('[ training... ]')
        self.parleys_episode = 0
        self.max_num_epochs = opt['num_epochs'] if opt['num_epochs'] > 0 else float('inf')
        self.max_train_time = opt['max_train_time'] if opt['max_train_time'] > 0 else float('inf')
        self.log_every_n_secs = opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 else float('inf')
        self.train_dis_every_n_secs = opt['train_display_every_n_secs'] if opt['train_display_every_n_secs'] > 0 else float('inf')
        self.val_every_n_secs = opt['validation_every_n_secs'] if opt['validation_every_n_secs'] > 0 else float('inf')
        self.save_every_n_secs = opt['save_every_n_secs'] if opt['save_every_n_secs'] > 0 else float('inf')
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.best_valid = None
        if opt.get('model_file_transmitter') and os.path.isfile(opt['model_file_transmitter'] + '.best_valid'):
            with open(opt['model_file_transmitter'] + ".best_valid", 'r') as f:
                x = f.readline()
                self.best_valid = float(x)
                f.close()
        self.impatience = 0
        self.saved = False
        self.valid_world = None
        self.opt = opt
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)

    def validate(self):
        opt = self.opt
        valid_report, self.valid_world = run_eval(
            self.agent, opt, 'valid', opt['validation_max_exs'],
            valid_world=self.valid_world)
        if opt['tensorboard_log'] is True:
            self.writer.add_metrics('valid', self.parleys_episode, valid_report)
        if opt.get('model_file_transmitter') and opt.get('save_after_valid'):
            print("[ saving transmitter checkpoint: " + opt['model_file_transmitter'] + ".checkpoint ]")
            self.agent.save(component='transmitter')
        # if opt.get('model_file_receiver') and opt.get('save_after_valid'):
        #     print("[ saving receiver checkpoint: " + opt['model_file_receiver'] + ".checkpoint ]")
        #     self.agent.save(component='receiver')
        if hasattr(self.agent, 'receive_metrics'):
            self.agent.receive_metrics(valid_report)
        if '/' in opt['validation_metric']:
            # if you are multitasking and want your validation metric to be
            # a metric specific to a subtask, specify your validation metric
            # as -vmt subtask/metric
            subtask = opt['validation_metric'].split('/')[0]
            validation_metric = opt['validation_metric'].split('/')[1]
            new_valid = valid_report['tasks'][subtask][validation_metric]
        else:
            new_valid = valid_report[opt['validation_metric']]
        if self.best_valid is None or self.valid_optim * new_valid > self.valid_optim * self.best_valid:
            print('[ new best {}: {}{} ]'.format(
                opt['validation_metric'], new_valid,
                ' (previous best was {})'.format(self.best_valid)
                    if self.best_valid is not None else ''))
            self.best_valid = new_valid
            self.impatience = 0
            if opt.get('model_file_transmitter'):
                print("[ saving best valid model: " + opt['model_file_transmitter'] + " ]")
                self.agent.save(component='transmitter')
                print("[ saving best valid metric: " + opt['model_file_transmitter'] + ".best_valid ]")
                save_best_valid(opt['model_file_transmitter'], self.best_valid)
                self.saved = True

            if opt['validation_metric'] == 'accuracy' and self.best_valid >= opt['validation_cutoff']:
                print('[ task solved! stopping. ]')
                return True
        else:
            self.impatience += 1
            print('[ did not beat best {}: {} impatience: {} ]'.format(
                    opt['validation_metric'], round(self.best_valid, 4),
                    self.impatience))
        self.validate_time.reset()
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
        logs.append('parleys:{}'.format(self.parleys_episode))

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
            self.writer.add_metrics('train', self.parleys_episode, train_report)

    def train(self):
        # print('#### Validating at {} training episode '.format(self.parleys_episode))
        # self.validate()
        opt = self.opt
        world = self.world
        with world:
            while True:
                self.parleys_episode += 1
                if self.parleys_episode % 100 == 0:
                    print('#### Training {} episode '.format(self.parleys_episode))

                if self.train_dis_time.time() > self.train_dis_every_n_secs:
                    is_display = True
                    # clear to zero
                    self.train_dis_time.reset()
                else:
                    is_display = False

                world.parley_episode(is_training=True, is_display=is_display)

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
                    print('#### Validating at {} training episode '.format(self.parleys_episode))
                    stop_training = self.validate()
                    if stop_training:
                        break

                if self.save_time.time() > self.save_every_n_secs:
                    if opt.get('model_file_transmitter'):
                        print("[ saving transmitter checkpoint: " + opt['model_file_transmitter'] + ".checkpoint ]")
                        self.agent.save(opt['model_file_transmitter'] + '.checkpoint', component='transmitter')
                    if opt.get('model_file_receiver'):
                        print("[ saving receiver checkpoint: " + opt['model_file_receiver'] + ".checkpoint ]")
                        self.agent.save(opt['model_file_receiver'] + '.checkpoint', component='receiver')
                    self.save_time.reset()

        if not self.saved:
            # save agent
            self.agent.save(component='transmitter')
            # self.agent.save(component='receiver') # TODO: API for save all components
        elif opt.get('model_file_transmitter') and opt.get('model_file_receiver'): # TODO: check if both components are necessary
            # reload best validation model
            self.agent = create_agent(opt)

        v_report, v_world = run_eval(self.agent, opt, 'valid', write_log=True)
        t_report, t_world = run_eval(self.agent, opt, 'test', write_log=True)
        v_world.shutdown()
        t_world.shutdown()
        return v_report, t_report


if __name__ == '__main__':
    TrainLoop(setup_args().parse_args()).train()
    print()
