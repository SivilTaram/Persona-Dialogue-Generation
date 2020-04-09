"""Self-play between two user agents with initial message provided by the task_agents"""
from parlai.core.agents import _create_task_agents
from parlai.core.worlds import DialogPartnerWorld, BatchWorld
import numpy as np
from parlai.core.worlds import create_agents_from_shared
import random
from copy import deepcopy
import torch
import math
import torch.nn.functional as F


def validate(observation):
    """Make sure the observation table is valid, or raise an error."""
    if observation is not None and type(observation) == dict:
        return observation
    elif type(observation) == list:
        return observation
    else:
        raise RuntimeError('Must return dictionary from act().')


def information_penalty(vt_agent):
    norm = vt_agent.norm(p=2, dim=2, keepdim=True) + 1e-3
    utt_sim_mat = torch.matmul(vt_agent, vt_agent.transpose(1, 2)) / torch.matmul(norm,
                                                                                  norm.transpose(1, 2))
    utt_size = utt_sim_mat.size(-1)
    self_mask = torch.tril(torch.ones((utt_size, utt_size), device=utt_sim_mat.device), diagonal=-1)
    mask_sim_mat = utt_sim_mat * self_mask
    # mean on one dimension TODO: mean or sum ?
    inform_flow_penalty = torch.max(mask_sim_mat, dim=2)[0]
    return inform_flow_penalty


class SelfPlayWorld(DialogPartnerWorld):
    """Simple world where two user agents practice communicating with the initial messages
    provided by the task_agents at the beginning of each episode
    """

    # TODO: check if agent[0]'s act has ['episode_done']
    # TODO: check if there is total_episodes in opt
    def __init__(self, opt, agents, shared=None):
        """

        :param opt:
        :param agents: list of agents, [user_agent_a, user_agent_b, task_agent]
                       task_agent provide initial information e.g. persona
                       user_agent_a, user_agent_b have a conversation according
                       to given persona
        :param shared:
        """
        # last two are actual agents
        super().__init__(opt, agents=agents[:2], shared=None)
        self.super_teacher = agents[2] if len(agents) >= 3 else None
        self.self_play_teacher = agents[3] if len(agents) >= 4 else None

        self.act_id = 0  # act_id in current episode
        self.episode_id = 0  # episode_id

        # default is 5
        self.sample_num = opt.get('sample_num', 5)
        self.reward_sum = 0.0
        self.reward_count = 0.0
        self.super_batchsize = opt['batchsize']

        self.running_baseline = np.zeros(self.opt['max_turn'])
        self.running_rounds = 0
        self.current_turn = 3

    def _set_agent_mode(self, is_training, is_display):
        for agent in self.agents:
            # reset agent and begin training
            agent.reset()
            agent.set_mode(is_training)
        # select one as user
        self.agents[1].set_greedy()
        # if random.random() > 0.5:
        #     if is_display:
        #         print("[ A is the user ]")
        #     self.agents[0].set_greedy()
        # else:
        #     if is_display:
        #         print("[ B is the user ]")
        #     self.agents[1].set_greedy()

    def parley(self, is_display=False):
        """Agent 0 goes first. Alternate between the two agents."""
        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act(is_display)
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act(is_display)
        agents[0].observe(validate(acts[1]))
        self.update_counters()

    def parley_episode(self, is_training=True, is_display=False):
        # agent select: Teacher or SelfPlay
        self._set_agent_mode(is_training, is_display)

        use_supervision = random.random() > 0.9

        if use_supervision:
            agents = self.agents
            # batch observations can be split into two list next
            batch_observations = []

            # gather batch observations
            temp_observation = []
            max_dialog_len = 0
            for i in range(self.super_batchsize):
                agent_super_obs = self.super_teacher.act()
                while not agent_super_obs['episode_done']:
                    temp_observation.append(agent_super_obs)
                    agent_super_obs = self.super_teacher.act()

                batch_observations.append(temp_observation)
                max_dialog_len = max(max_dialog_len, len(temp_observation))
                temp_observation = []

            # step for longest dialogue in batch
            for obs in batch_observations:
                cur_dialog_len = len(obs)
                if cur_dialog_len < max_dialog_len:
                    # fill in
                    add_step = max_dialog_len - cur_dialog_len
                    for i in range(add_step):
                        # from the head to tail, repeat until equal length
                        obs.append(deepcopy(obs[i]))
            # transpose it into batch first
            batch_observations = [list(x) for x in zip(*batch_observations)]

            for batch_obs in batch_observations:
                # super teach has been acted
                agents[0].observe(batch_obs)
                agents[0].act()
                # TODO: current super teacher cannot observe
                # self.super_teacher.observe(validate(acts[0]))
                agents[0].update_supervised()
        else:
            agents = self.agents
            # while True:
            # for i in range(self.opt['batchsize']):
            obs_start_a = self.self_play_teacher.act()  # persona for agent a
            obs_start_a['episode_done'] = False  # ensure persona is added to history
            obs_start_b = self.self_play_teacher.act()  # persona for agent b
            obs_start_b['episode_done'] = False  # ensure persona is added to history

            # cut the persona and use another's text as first sentence
            a_text = obs_start_a['text']
            sep_a = a_text.rfind('\n')
            b_text = obs_start_b['text']
            sep_b = b_text.rfind('\n')
            persona_a_text, first_mess_a = a_text[:sep_a], a_text[sep_a:]
            persona_b_text, first_mess_b = b_text[:sep_b], b_text[sep_b:]

            # reconstruct
            # if random.random() > 0.5:
            #     first_mess_b = '\n__SILENCE__'

            obs_start_a['text'] = persona_a_text + first_mess_b
            obs_start_b['text'] = persona_b_text
            # start_obs_a.append(obs_start_a)
            # start_obs_b.append(obs_start_b)
            # first_message.append(first_mess_b)
            agents[0].observe([deepcopy(obs_start_a) for _ in range(self.opt['batchsize'])])
            agents[1].observe([deepcopy(obs_start_b) for _ in range(self.opt['batchsize'])])
            # agents[0].observe(deepcopy(start_obs_a))
            # agents[1].observe(deepcopy(start_obs_b))

            first_message = [first_mess_b] * self.opt['batchsize']

            # actual persona are confess by SELF
            persona_agent_a = agents[0].confess()
            persona_agent_b = agents[1].confess()

            if is_display:
                print('```')
                print(' [ Episode starts! Allocate persona for both interlocutors ...] ')
                print('A: ' + obs_start_a['text'])
                print('B: ' + obs_start_b['text'])

            # reset score sum as zero
            if is_display:
                print('---------------- Dialogue ------------------')
            while not self.episode_done():
                self.parley(is_display)

            # virtual persona are understand by SELF
            vt_persona_agent_a = agents[0].understand()
            vt_persona_agent_b = agents[1].understand()

            # receive message and send message are as attribute of the agent
            # therefore we here only pass-by the first message
            agent_a_coherent_reward = agents[0].coherent_score(first_message)
            agent_b_coherent_reward = agents[1].coherent_score(first_message)

            agent_a_language_reward = agents[0].language_score()
            agent_b_language_reward = agents[1].language_score()

            # get batch size
            batch_size = vt_persona_agent_a.size(0)

            persona_agent_a = persona_agent_a.unsqueeze(dim=0).repeat(batch_size, 1, 1)
            persona_agent_b = persona_agent_b.unsqueeze(dim=0).repeat(batch_size, 1, 1)

            agent_a_persona_reward = agents[0].measure_persona_similarity(vt_persona_agent_a, persona_agent_a)
            agent_b_persona_reward = agents[1].measure_persona_similarity(vt_persona_agent_b, persona_agent_b)

            turn_size = agent_a_persona_reward.shape[1]

            # delayed reward
            discount_gamma = 0.5

            if discount_gamma > 0:
                # use first second to reorder
                if agents[0].is_first_speaker:
                    history_reward = np.stack([agent_a_persona_reward, agent_b_persona_reward], axis=2)
                else:
                    history_reward = np.stack([agent_b_persona_reward, agent_a_persona_reward], axis=2)
                # view as a whole
                history_reward = history_reward.reshape((batch_size, -1))
                discount_rewards = []

                dialog_size = len(history_reward[0])
                # reward of each step
                step_reward = np.array([0.0 for _ in range(batch_size)])
                for i in reversed(range(dialog_size)):
                    r = history_reward[:, i]
                    step_reward = r + discount_gamma * step_reward
                    discount_rewards.insert(0, step_reward)
                # persona rewards reallocate
                discount_rewards = np.array(discount_rewards)
                # discount_rewards as step_size x batch_size
                discount_rewards = np.transpose(discount_rewards)
                discount_rewards = discount_rewards.reshape((batch_size, turn_size, 2))

                agent_a_persona_reward = discount_rewards[:, :, 0]
                agent_b_persona_reward = discount_rewards[:, :, 1]

                if not agents[0].is_first_speaker:
                    # reverse
                    agent_a_persona_reward, agent_b_persona_reward = agent_b_persona_reward, agent_a_persona_reward

            agent_person_reward = np.concatenate([agent_a_persona_reward, agent_b_persona_reward], axis=1)
            # min-max normalization
            min_persona_reward = agent_person_reward.min()
            max_persona_reward = agent_person_reward.max()
            diff_reward = max_persona_reward - min_persona_reward + 1e-6
            agent_person_reward = 2 * (agent_person_reward - min_persona_reward) / diff_reward
            turn_size = agent_a_persona_reward.shape[1]
            agent_a_persona_reward = agent_person_reward[:, :turn_size]
            agent_b_persona_reward = agent_person_reward[:, turn_size:]

            # define information penalty
            if agents[0].is_first_speaker:
                information_agent = torch.stack((vt_persona_agent_a, vt_persona_agent_b), dim=2)
            else:
                information_agent = torch.stack((vt_persona_agent_b, vt_persona_agent_a), dim=2)

            information_agent = information_agent.view(batch_size, turn_size * 2, -1)
            information_agent_penalty = information_penalty(information_agent)
            information_agent_penalty = information_agent_penalty.view(batch_size, 2, turn_size).data.cpu().numpy()
            information_penalty_a = information_agent_penalty[:, 0, :]
            information_penalty_b = information_agent_penalty[:, 1, :]

            if not agents[0].is_first_speaker:
                information_penalty_a, information_penalty_b = information_penalty_b, information_penalty_a
            # as one score or multi score
            reward_a_list = 0.1 * agent_a_coherent_reward + 0.5 * agent_a_persona_reward + 0.5 * agent_a_language_reward - 0.1 * information_penalty_a
            reward_b_list = 0.1 * agent_b_coherent_reward + 0.5 * agent_b_persona_reward + 0.5 * agent_b_language_reward - 0.1 * information_penalty_b

            # subtract running average
            reward_a_baseline = reward_a_list.mean(axis=0, keepdims=True)
            reward_a_list = reward_a_list - reward_a_baseline

            reward_b_baseline = reward_b_list.mean(axis=0, keepdims=True)
            reward_b_list = reward_b_list - reward_b_baseline

            if is_display:
                print('---------------- Reward A ------------------')
                print('coherent : {}'.format(agent_a_coherent_reward[0]))
                print('language : {}'.format(agent_a_language_reward[0]))
                print('persona  : {}'.format(agent_a_persona_reward[0]))
                print('penalty  : {}'.format(information_penalty_a[0]))
                print('---------------- Reward B ------------------')
                print('coherent : {}'.format(agent_b_coherent_reward[0]))
                print('language : {}'.format(agent_b_language_reward[0]))
                print('persona  : {}'.format(agent_b_persona_reward[0]))
                print('penalty  : {}'.format(information_penalty_b[0]))

            # Send end signal to agent a and agent b
            # Here reward is defined as a list, not a log probability
            obs_end_a = [{'id': agents[0].id, 'episode_done': True, 'reward': reward_a} for reward_a in reward_a_list]
            obs_end_b = [{'id': agents[1].id, 'episode_done': True, 'reward': reward_b} for reward_b in reward_b_list]
            agents[0].observe(obs_end_a)
            agents[1].observe(obs_end_b)

            agents[0].update_selfplay()
            # update counters only apply on rl training
            self.update_counters()

    def update_counters(self):
        """Update various counters"""
        if self.episode_done():
            self.num_act_cur_episode = 0  # reset to zero after an episode is done
            self.num_episodes += 1
            return None
        else:
            self.num_act_cur_episode += 1

        self.total_parleys += 1
        if self.max_exs is None:
            if ('num_epochs' in self.opt and self.opt['num_epochs'] > 0):
                self.max_exs = self.num_examples() * self.opt['num_epochs'] if self.num_examples() else -1
            else:
                self.max_exs = -1

        if self.max_exs > 0:
            self.total_epochs = self.total_parleys * self.opt.get('batchsize', 1) / self.num_examples()
        else:
            if self.epoch_done():
                self.total_epochs += 1
        # every 400 parleys, increasing 1
        if self.total_parleys > (self.current_turn * 500 - 500) and self.current_turn < self.opt.get('max_turn', 5):
            self.current_turn += 1

    @property
    def num_act_cur_episode(self):
        """Number of actions in current episode"""
        return self.act_id

    @num_act_cur_episode.setter
    def num_act_cur_episode(self, v):
        self.act_id = v

    @property
    def num_episodes(self):
        """Number of episodes up to now"""
        return self.episode_id

    @num_episodes.setter
    def num_episodes(self, v):
        self.episode_id = v

    def num_examples(self):
        return self.self_play_teacher.num_examples()  # TODO: check originalpersonateacher has this function

    def episode_done(self):
        if self.num_act_cur_episode >= self.current_turn:
            return True
        else:
            return False


def create_selfplay_world(opt, user_agents):
    """API compatible for various selfplay task"""
    # TODO: tasks.convai2.agents.OriginalPersonaTeacher, add more teachers
    if ',' in opt['task']:
        tasks = opt['task'].split(',')
        teacher_agents = []
        for task in tasks:
            opt['task'] = task
            teacher_agents.append(_create_task_agents(opt)[0])
    else:
        teacher_agents = [_create_task_agents(opt)[0]]

    agents = [user_agents[0], user_agents[1]]
    agents.extend(teacher_agents)
    world = SelfPlayWorld(opt, agents=agents, shared=None)
    return world
