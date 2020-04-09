import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel


class Gpt2SeqModel(nn.Module):
    def __init__(self, opt, vocab_size, pad_idx, start_idx, end_idx, special_token_len, dict, longest_label=1,
                 length_penalty=1.0, diversity_groups=1, diversity_coef=0.2, annealing_topk=None, annealing=0, sample=False,
                 temperature=0.7):
        super().__init__()
        # original vocab size plus special vocab
        self.vocab_size = vocab_size + 40478
        self.token_type_dict = {}
        # max is 30
        for i in range(29):
            self.token_type_dict['dis'+str(i)] = self.vocab_size + i
        # pred for prediction turn embedding
        self.token_type_dict['pred'] = self.vocab_size + 29
        # the remaining 30 is the distance size
        special_token_len += 30
        self.vocab_size += 29
        # regard input and output as one sentence, given the input as context, generate the next sentence.
        self.transformer_module = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt',
                                                                       num_special_tokens=special_token_len)
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.register_buffer('start_tensor', torch.LongTensor([start_idx]))
        self.register_buffer('pred_turn_tensor', torch.LongTensor([self.token_type_dict['pred']]))
        # default beam equal to 1
        self.beam_size = opt.get('beam_size', 1)
        self.rank = opt.get('rank_candidates', False)

        self.use_turn = opt.get('encoder_turn_use', False)
        self.use_dis = opt.get('encoder_dis_use', False)
        # longest label
        self.longest_label = min(longest_label, opt.get('decode_max_seq_len', 100))
        self.length_penalty_coef = length_penalty
        self.diversity_groups = diversity_groups
        self.diversity_coef = diversity_coef
        self.annealing_topk = annealing_topk
        self.annealing = annealing
        self.temperature = temperature
        self.topk = opt.get('top_k', 0)
        self.dict = dict
        self.no_repeat_ngram_size = 2
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(768, 2, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, src_seq, src_seq_turn=None, src_seq_dis=None, tgt_seq=None, tgt_seq_turn=None, cands=None, valid_cands=None, prev_enc=None,
                rank_during_training=False, sampling=False, sampling_cands=None):
        # concat src_seq and tgt_seq as one sentence, use start token to separate them.
        if tgt_seq is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, tgt_seq.size(1))

        batch_size, src_seq_len = src_seq.size()

        if src_seq_dis is not None:
            src_seq_dis = np.array(src_seq_dis)

        # evaluation return none scores
        scores = None

        negative_score = None
        start_tensor = self.start_tensor.detach().expand(batch_size, 1)
        # whether training or evaluation
        if tgt_seq is not None:
            input_seq = torch.cat([src_seq, start_tensor, tgt_seq], dim=1)
            # TODO: manually construct the position ids for input & output
            if self.use_dis and src_seq_dis is not None:
                # create numpy
                token_type_ids = np.zeros(input_seq.size())
                # map str to id
                for row_ind, row in enumerate(token_type_ids):
                    for ind, _ in enumerate(row):
                        if ind < src_seq_len:
                            str_ind = 'dis' + str(src_seq_dis[row_ind, ind])
                            row[ind] = self.token_type_dict[str_ind]
                        else:
                            row[ind] = self.token_type_dict['pred']
                dis_seq = torch.tensor(token_type_ids, device=input_seq.device, dtype=torch.long)
            else:
                dis_seq = None
            lm_logits, hidden_states = self.transformer_module(input_seq, None, dis_seq)
            # lm labels should mask the source sentence language model
            shift_logits = lm_logits[..., src_seq_len:-1, :].contiguous()
            # lm_labels = tgt_seq.clone()[..., 1:].contiguous()
            # predict answers
            scores = shift_logits
            pos_seq_len = src_seq_len + tgt_seq.ne(self.pad_idx).sum(dim=1, keepdim=True)
            pos_seq_len_expand = pos_seq_len.unsqueeze(dim=2).repeat(1, 1, 768)
            last_state = hidden_states.gather(dim=1, index=pos_seq_len_expand).squeeze(dim=1)
            positive_score = self.linear(self.dropout(last_state))
            predictions = shift_logits.argmax(dim=-1)
        else:
            prior_context = torch.cat([src_seq, start_tensor], dim=1)
            if self.use_dis and src_seq_dis is not None:
                # create numpy
                token_type_ids = np.zeros(prior_context.size())
                # map str to id
                for row_ind, row in enumerate(token_type_ids):
                    for ind, _ in enumerate(row):
                        if ind < src_seq_len:
                            str_ind = 'dis' + str(src_seq_dis[row_ind, ind])
                            row[ind] = self.token_type_dict[str_ind]
                        else:
                            row[ind] = self.token_type_dict['pred']
                prior_dis = torch.tensor(token_type_ids, device=prior_context.device, dtype=torch.long)
            else:
                prior_dis = None

            if sampling:
                predictions, scores, hidden_states = self.sample_decoding(batch_size, prior_context, prior_dis, self.topk)
            elif self.training:
                predictions, scores, hidden_states = self.train_greedy_decoding(batch_size, prior_context, prior_dis)
            elif self.beam_size > 1:
                predictions, hidden_states = self.beam_search(batch_size, prior_context)
            else:
                predictions, hidden_states = self.greedy_decoding(batch_size, prior_context, prior_dis)

            positive_score = self.linear(hidden_states[:, -1, :])

        if self.training and sampling_cands is not None:
            sampling_seq = sampling_cands
            sampling_seq_len = src_seq_len + sampling_seq.ne(self.pad_idx).sum(dim=1, keepdim=True)
            sampling_seq_len_expand = sampling_seq_len.unsqueeze(dim=2).repeat(1, 1, 768)

            cand_seq = torch.cat([src_seq, start_tensor, sampling_seq], dim=1)
            # TODO: manually construct the position ids for input & output
            sampling_logits, hidden_states = self.transformer_module(cand_seq, None, None)
            # lm labels should mask the source sentence language model
            last_state = hidden_states.gather(dim=1, index=sampling_seq_len_expand).squeeze(dim=1)
            negative_score = self.linear(self.dropout(last_state))

        cand_preds, cand_scores = None, None

        if self.rank and cands is not None and len(cands) > 0:
            if (self.training and rank_during_training) or (not self.training):
                # candidates are all target sequence
                cand_scores = []
                # TODO: we will reorder the sequence at transmitter
                with torch.no_grad():
                    for ind in range(len(cands)):
                        current_cs = cands[ind]
                        cand_size = current_cs.size()[0]
                        # repeat src sequence on dimension 1
                        cand_src_seq = src_seq[ind].detach().expand(cand_size, -1)
                        # creat start tensor
                        cand_start = self.start_tensor.detach().expand(cand_size, 1)

                        # concat them into one
                        cand_inp_seq = torch.cat([cand_src_seq, cand_start, current_cs], dim=1).contiguous()
                        cand_logits, hidden_states = self.transformer_module(cand_inp_seq)
                        # view as batch_size x cand_size
                        # TODO: note the candidate doesn't own a END symbol,
                        #  so we ignore the score form last word -> EOS and EOS -> pad
                        cand_logits = cand_logits[..., src_seq_len:-1, :].contiguous()
                        # TODO: start -> I, I -> love, ... the -> world, [world -> EOS] (missing here), target is the
                        #  original sentence
                        cand_label = current_cs
                        true_score = F.log_softmax(cand_logits, dim=2).gather(2, cand_label.unsqueeze(2))
                        nonzero = current_cs.ne(self.pad_idx).float()
                        current_cs_score = (true_score.squeeze(2) * nonzero).sum(1)
                        current_cs_lens = nonzero.sum(1)
                        current_cs_score /= current_cs_lens  # **len_penalty?

                        current_cs_valid_len = src_seq_len + nonzero.sum(dim=1, keepdim=True).long()
                        current_cs_valid_len = current_cs_valid_len.unsqueeze(dim=2).repeat(1, 1, 768)
                        last_state = hidden_states.gather(dim=1, index=current_cs_valid_len).squeeze(dim=1)
                        # 1 is pos, 0 is neg
                        current_rank_score = F.softmax(self.linear(last_state), dim=1)[:, 1]
                        current_cs_score = 1.0 * current_rank_score
                        cand_scores.append(current_cs_score.view(1, -1))

                    cand_scores = torch.cat(cand_scores, dim=0)
                    cand_preds = cand_scores.sort(1, True)[1]

        return predictions, scores, cand_preds, cand_scores, positive_score, negative_score

    # TODO: we do not do any penalty
    def _length_penalty(self, sequence_lengths):
        return sequence_lengths

    def greedy_decoding(self, batch_size, prior_context, prior_dis):
        device = next(self.parameters()).device
        # predict_tok = torch.full((batch_size, 1), fill_value=self.start_idx, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, dtype=torch.uint8, device=device)

        pred_output = torch.zeros((batch_size, self.longest_label), dtype=torch.long, device=device)
        past_input = prior_context
        past_dis = prior_dis
        with torch.no_grad():
            for step in range(self.longest_label):
                logits, hidden_states = self.transformer_module.forward(past_input, None, past_dis)
                # logits, _ = self.transformer_module.forward(past)
                logits = logits[:, -1, :] / self.temperature
                log_probs = F.log_softmax(logits, dim=-1)

                if 0 < self.no_repeat_ngram_size < step:
                    # for each beam and batch sentence, generate a list of previous ngrams
                    gen_ngrams = [{} for _ in range(batch_size)]
                    for bbsz_idx in range(batch_size):
                        gen_tokens = pred_output[bbsz_idx][pred_output[bbsz_idx].ne(self.pad_idx)].tolist()
                        for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                            gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

                predict_tok = torch.argmax(log_probs, dim=-1)
                # look forward one step
                pred_output[:, step] = predict_tok

                # mask banned repeated tokens
                if 0 < self.no_repeat_ngram_size < step:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(pred_output[bbsz_idx, step + 1 - self.no_repeat_ngram_size:step].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(batch_size)]
                    else:
                        banned_tokens = [[] for _ in range(batch_size)]

                    for bbsz_idx in range(batch_size):
                        log_probs[bbsz_idx, banned_tokens[bbsz_idx]] = -np.inf

                    predict_tok = torch.argmax(log_probs, dim=-1)
                    # look forward one step
                    pred_output[:, step] = predict_tok

                pred_output[is_end, step] = self.pad_idx
                past_input = torch.cat([past_input, predict_tok.view(batch_size, -1)], dim=1)
                if self.use_dis:
                    new_dis = self.pred_turn_tensor.repeat(batch_size).view(batch_size, -1)
                    past_dis = torch.cat([past_dis, new_dis], dim=1)
                is_end = is_end | (predict_tok == self.end_idx).view(-1)
                if (~is_end).sum() == 0:
                    break
        return pred_output, hidden_states

    def train_greedy_decoding(self, batch_size, prior_context, prior_dis):
        """
        This function is used to simulate the User in self-play.
        The only difference between this function with greedy_decoding is that
        this function will keep the gradient in the computing graph.
        """
        device = next(self.parameters()).device
        # predict_tok = torch.full((batch_size, 1), fill_value=self.start_idx, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, dtype=torch.uint8, device=device)

        pred_output = torch.zeros((batch_size, self.longest_label), dtype=torch.long, device=device)
        past_input = prior_context
        past_dis = prior_dis

        # start as the last element
        src_len = prior_context.size(1) - 1
        last_logits = None
        for step in range(self.longest_label):
            logits, hidden_states = self.transformer_module.forward(past_input, None, past_dis)
            last_logits= logits
            # logits, _ = self.transformer_module.forward(past)
            logits = logits[:, -1, :] / self.temperature
            # add score
            log_probs = F.softmax(logits, dim=-1)

            if 0 < self.no_repeat_ngram_size < step:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for _ in range(batch_size)]
                for bbsz_idx in range(batch_size):
                    gen_tokens = pred_output[bbsz_idx][pred_output[bbsz_idx].ne(self.pad_idx)].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            predict_tok = torch.argmax(log_probs, dim=-1)
            # look forward one step
            pred_output[:, step] = predict_tok

            # mask banned repeated tokens
            if 0 < self.no_repeat_ngram_size < step:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(pred_output[bbsz_idx, step + 1 - self.no_repeat_ngram_size:step].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(batch_size)]
                else:
                    banned_tokens = [[] for _ in range(batch_size)]

                for bbsz_idx in range(batch_size):
                    log_probs[bbsz_idx, banned_tokens[bbsz_idx]] = -np.inf

                predict_tok = torch.argmax(log_probs, dim=-1)
                # look forward one step
                pred_output[:, step] = predict_tok

            pred_output[is_end, step] = self.pad_idx
            past_input = torch.cat([past_input, predict_tok.view(batch_size, -1)], dim=1)
            if self.use_dis:
                new_dis = self.pred_turn_tensor.repeat(batch_size).view(batch_size, -1)
                past_dis = torch.cat([past_dis, new_dis], dim=1)
            is_end = is_end | (predict_tok == self.end_idx).view(-1)
            if (~is_end).sum() == 0:
                break
        score_output = last_logits[..., src_len:, :].contiguous()
        pred_output = pred_output[..., :score_output.shape[1]].contiguous()
        return pred_output, score_output, hidden_states

    def sample_decoding(self, batch_size, prior_context, prior_dis, topk):
        """
        This function is used to simulate the Learned Agent in self-play
        The parameter topk specifies the sampling space at each decoding step.
        """
        device = next(self.parameters()).device
        # predict_tok = torch.full((batch_size, 1), fill_value=self.start_idx, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, dtype=torch.uint8, device=device)

        pred_output = torch.zeros((batch_size, self.longest_label), dtype=torch.long, device=device)
        past_input = prior_context
        past_dis = prior_dis

        # start as the last element
        src_len = prior_context.size(1) - 1
        last_logits = None
        for step in range(self.longest_label):
            logits, hidden_states = self.transformer_module.forward(past_input, None, past_dis)
            last_logits= logits
            # logits, _ = self.transformer_module.forward(past)
            logits = logits[:, -1, :] / self.temperature
            # add score
            logits = top_k_logits(logits, k=topk)
            probs = F.softmax(logits, dim=-1)

            if 0 < self.no_repeat_ngram_size < step:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for _ in range(batch_size)]
                for bbsz_idx in range(batch_size):
                    gen_tokens = pred_output[bbsz_idx][pred_output[bbsz_idx].ne(self.pad_idx)].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                            gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            predict_tok = torch.multinomial(probs, num_samples=1)
            # must one (including END)
            # look forward one step
            pred_output[:, step] = predict_tok.view(-1)

            # mask banned repeated tokens
            if 0 < self.no_repeat_ngram_size < step:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(pred_output[bbsz_idx, step + 1 - self.no_repeat_ngram_size:step].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(batch_size)]
                else:
                    banned_tokens = [[] for _ in range(batch_size)]

                for bbsz_idx in range(batch_size):
                    probs[bbsz_idx, banned_tokens[bbsz_idx]] = 0.0

                predict_tok = torch.multinomial(probs, num_samples=1)
                # look forward one step
                pred_output[:, step] = predict_tok.view(-1)

            pred_output[is_end, step] = self.pad_idx
            past_input = torch.cat([past_input, predict_tok], dim=1)
            if self.use_dis:
                new_dis = self.pred_turn_tensor.repeat(batch_size).view(batch_size, -1)
                past_dis = torch.cat([past_dis, new_dis], dim=1)
            is_end = is_end | (predict_tok == self.end_idx).view(-1)
            if (~is_end).sum() == 0:
                break
        score_output = last_logits[..., src_len:, :].contiguous()
        pred_output = pred_output[..., :score_output.shape[1]].contiguous()
        return pred_output, score_output, hidden_states

    def beam_search(self, batch_size, prior_context):
        """
        beam search for the validating generation. Note we also impose the n-gram repeating, which is borrowed
        from https://github.com/pytorch/fairseq. The diversity is not useful here.
        """
        device = next(self.parameters()).device

        beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
        beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

        current_sample_prob = 1
        group_size = self.beam_size // self.diversity_groups
        diversity_penalty = torch.zeros((batch_size, self.vocab_size), device=device)

        prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.start_idx, dtype=torch.long,
                           device=device)
        with torch.no_grad():
            # initialize presents
            start_token_tensor = prior_context.repeat(1, self.beam_size).view(batch_size * self.beam_size, -1)
            token_tensor = start_token_tensor
            for step in range(self.longest_label):
                outputs, hidden_states = self.transformer_module.forward(token_tensor)

                logits = outputs[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)

                if 0 < self.no_repeat_ngram_size < step:
                    # for each beam and batch sentence, generate a list of previous ngrams
                    gen_ngrams = [{} for bbsz_idx in range(batch_size * self.beam_size)]
                    for bbsz_idx in range(batch_size * self.beam_size):
                        gen_tokens = prevs[bbsz_idx].tolist()
                        for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                            gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

                if 0 < self.no_repeat_ngram_size < step:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = tuple(prevs[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                        return gen_ngrams[bbsz_idx].get(ngram_index, [])

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(batch_size * self.beam_size)]
                    else:
                        banned_tokens = [[] for bbsz_idx in range(batch_size * self.beam_size)]

                    for bbsz_idx in range(batch_size * self.beam_size):
                        log_probs[bbsz_idx, banned_tokens[bbsz_idx]] = log_probs[bbsz_idx, banned_tokens[bbsz_idx]] * 100

                log_probs = log_probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, self.vocab_size)
                beam_scores = beam_scores / penalty

                if step == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            beam_probas = F.softmax(g_beam_scores, dim=-1)
                            if self.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size, replacement=True)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * self.vocab_size

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.vocab_size),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.vocab_size).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.pad_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.end_idx] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing
                token_tensor = torch.cat([start_token_tensor, prevs], dim=1)

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            bests = beam_scores.argmax(dim=-1)

            for step in range(batch_size):
                best_len = beam_lens[step, bests[step]]
                best_seq = result[step, bests[step], 1:best_len - 1]
                predicts.append(best_seq.tolist())

        return predicts, hidden_states

    def score_sentence(self, receive_tokens, send_tokens):
        """
        Here we assume the generate tokens have containing the first message from sampling
        :param receive_tokens: tokens from others in round N
        :param send_tokens: tokens sent from self in round N
        :return:
        """
        batch_size, turn_size, prev_len = receive_tokens.size()
        start_tensor = self.start_tensor.detach().expand(batch_size, turn_size, 1)

        # clip the token sequence into maximum length if it is larger than 512
        max_len = 512 - 1 - send_tokens.size(-1)
        cur_token_len = receive_tokens.size(-1)

        assert receive_tokens.size(1) == send_tokens.size(1)
        if cur_token_len > max_len:
            receive_tokens = receive_tokens[:, :, cur_token_len-max_len:]

        # TODO: send_tokens should be appended end_token for classification
        all_tokens = torch.cat((receive_tokens, start_tensor, send_tokens), dim=2).contiguous()

        sen_size = all_tokens.size(-1)
        all_tokens = all_tokens.view(batch_size * turn_size, sen_size)
        # TODO: manually construct the position ids for input & output
        with torch.no_grad():
            lm_logits, hidden_states = self.transformer_module.forward(all_tokens)
            send_len = prev_len + send_tokens.ne(self.pad_idx).sum(dim=2, keepdim=True).view(batch_size * turn_size, -1)
            send_len_expand = send_len.unsqueeze(dim=2).repeat(1, 1, 768)
            # lm labels should mask the source sentence language model
            last_state = hidden_states.gather(dim=1, index=send_len_expand).squeeze(dim=1)
            scores = F.softmax(self.linear(last_state))[:, 1]
        scores = scores.view(batch_size, turn_size)
        return scores


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)