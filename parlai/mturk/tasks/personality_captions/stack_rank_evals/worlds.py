# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.tasks.personality_captions.build import build as build_pc
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from task_config import task_config as config
import numpy as np
import time
import os
import pickle
import random
import json
import base64
from PIL import Image
from io import BytesIO

CHOOSER = 'Chooser'
ONBOARD_MSG = '\nWelcome! \
        When you are ready to begin, \
        click the "I am ready, continue" button below\n'
PICK_BEST_MSG = '\nImage number {}! Please take a look at the image, and select \
                 the comment you think is <b>more engaging \
                 (interesting, captivating, attention-grabbing).</b>\
                  Then, please explain why you chose that comment \
                  in the chat box below.'
PICK_WORST_MSG = 'Now, please pick the the comment you think is the <b>LEAST</b> \
                  engaging.'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
CHAT_ENDED_MSG = 'You are done with {} images. Thanks for your time! \nPlease \
        click <span style="color:blue"><b>Done with this HIT</b> </span> \
        button below to finish this HIT.'
WAITING_MSG = 'Please wait...'


def load_image(path):
    return Image.open(path).convert('RGB')


class ExampleGenerator(object):
    """Retrieve Example from Personality-Captions Dataset"""
    def __init__(self, opt):
        self.opt = opt
        handle = './examples_stack{}{}{}.pkl'.format(
            '_sandbox' if opt['is_sandbox'] else '',
            opt['compare_key_1'],
            opt['compare_key_2'])
        self.examples_idx_stack_path = os.path.join(os.getcwd(), handle)
        build_pc(opt)
        data_path = opt.get('eval_data_path')
        with open(data_path) as f:
            self.data = json.load(f)

        if os.path.exists(self.examples_idx_stack_path):
            with open(self.examples_idx_stack_path, 'rb') as handle:
                self.idx_stack = pickle.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = list(range(len(self.data)))
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_example(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        ex = self.data[idx]
        return (idx, ex)

    def push_example(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.examples_idx_stack_path, 'wb') as handle:
            pickle.dump(self.idx_stack, handle)


class RoleOnboardWorld(MTurkOnboardWorld):
    """A world that provides the appropriate instructions during onboarding"""
    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        super().__init__(opt, mturk_agent)

    def parley(self):
        onboard_msg = {
            'id': 'SYSTEM',
            'text': ONBOARD_MSG}

        onboard_msg['task_description'] = config['task_description']
        self.mturk_agent.observe(onboard_msg)

        act = self.mturk_agent.act(timeout=self.max_onboard_time)

        # timeout
        if act['episode_done'] or (('text' in act and
                                    act['text'] == TIMEOUT_MESSAGE)):
            self.episodeDone = True
            return

        if 'text' not in act:
            control_msg = {'id': 'SYSTEM',
                           'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True


class MTurkPersonalityCaptionsStackRankWorld(MultiAgentDialogWorld):
    """World where an agent observes 5 images and 2 comments about the images,
       and chooses the more engaging comment
    """
    def __init__(self, opt, agents=None, shared=None, world_tag='NONE'):
        self.turn_idx = 0
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.chat_done = False
        self.world_tag = world_tag
        self.max_resp_time = opt['max_resp_time']  # in secs
        super().__init__(opt, agents, shared)
        self.agents = agents
        self.agent = agents[0]
        self.data = []
        self.num_images = opt['num_images']
        self.ck1 = opt.get('compare_key_1')
        self.ck2 = opt.get('compare_key_2')
        self.show_personality = opt.get('show_personality')
        if opt.get('yfcc_path'):
            self.image_path = opt['yfcc_path']
        else:
            self.image_path = os.path.join(opt['datapath'], 'yfcc_images')

    def episode_done(self):
        return self.chat_done

    def parley(self):
        """CHOOSER is given an image and 2 comments/captions, and is asked
           to choose more engaging comment.
        """
        # Initial Message Value
        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        """We only have to worry about 1 agent"""
        agent = self.agents[0]

        """First, we give CHOOSER the image
        """
        while self.turn_idx < self.num_images:
            print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))
            # Send image to turker
            control_msg['description'] = config['task_description']
            self.example_num, example = self.agent.example_generator.pop_example()
            img = load_image(os.path.join(
                             self.image_path,
                             '{}.jpg'.format(example['image_hash'])))
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            encoded = str(base64.b64encode(buffered.getvalue()).decode('ascii'))
            control_msg['image'] = encoded
            """
                Setup comments for ranking
            """
            comments = [(ck, example[ck]) for ck in [self.ck1, self.ck2]]
            random.shuffle(comments)
            control_msg['comments'] = [c[1] for c in comments]
            if self.show_personality:
                control_msg['personality'] = '<b><span style="color:blue">' \
                                               '{}\n</span></b>'.format(
                                              example['personality'].strip())

            best_pick = None
            control_msg['text'] = PICK_BEST_MSG.format(self.turn_idx + 1)
            control_msg['new_eval'] = True
            agent.observe(validate(control_msg))
            time.sleep(1)
            act = agent.act(timeout=self.max_resp_time)
            # First timeout check
            self.check_timeout(act)
            if self.chat_done:
                break
            try:
                best_idx = int(act['chosen'])
                reason = act['text']
                best_pick = comments[best_idx]
            except Exception:
                # Agent disconnected
                break
            example['choices'] = comments
            example['best_pick'] = best_pick
            example['reason'] = reason
            self.data.append(example)
            self.turn_idx += 1

        if self.turn_idx == self.num_images:
            control_msg['text'] = CHAT_ENDED_MSG.format(self.num_images)
            agent.observe(validate(control_msg))
        self.chat_done = True
        return

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = TIMEOUT_MSG
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        elif act['text'] == '[DISCONNECT]':
            self.chat_done = True
            return True
        else:
            return False

    def save_data(self):
        convo_finished = True
        for ag in self.agents:
            if (ag.hit_is_abandoned or ag.hit_is_returned or
                    ag.disconnected or ag.hit_is_expired):
                convo_finished = False
        if not convo_finished:
            ag.example_generator.push_example(self.example_num)
            print("\n**Push image {} back to stack. **\n".format(
                    self.example_num))
        self.agents[0].example_generator.save_idx_stack()
        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type))
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type))
        pickle.dump({'data': self.data,
                     'worker': self.agents[0].worker_id,
                     'hit_id': self.agents[0].hit_id,
                     'assignment_id': self.agents[0].assignment_id
                     }, open(filename, 'wb'))
        print('{}: Data successfully saved at {}.'.format(
            self.world_tag,
            filename))

    def review_work(self):
        global review_agent

        def review_agent(ag):
            pass  # auto approve 5 days
        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(review_agent)(agent) for agent in self.agents)

    def shutdown(self):
        """Shutdown all mturk agents in parallel, otherwise if one mturk agent
        is disconnected then it could prevent other mturk agents from
        completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            agent.shutdown()
        Parallel(
            n_jobs=len(self.agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.agents)
