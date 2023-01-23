import torch.nn as nn
import torch
import copy
import random
import numpy as np
import pdb
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import init


def list_split(l, ratio, shuffle=True):
    offset = int(len(l)*ratio)
    if shuffle:
        random.shuffle(l)
    return l[:offset], l[offset:]


def loss_func(loss, loss_type):
    if loss_type == 'margin':
        loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else 0
    elif loss_type == 'exp':
        loss = torch.mean(torch.exp(loss))
    elif loss_type == 'expmargin':
        loss = torch.mean(torch.exp(loss[loss > 0])) if min(loss[loss > 0].size()) > 0 else 0
    else:
        loss = None
        raise ValueError("loss type must be one of margin, exp and exp_margin, but now is {}".format(loss_type))
    return loss


def top_k_loss_func(loss, loss_type, tk=1000):
    loss = loss.contiguous().view(-1)
    if loss_type == 'margin':
        loss = loss[loss > 0] if min(loss[loss > 0].size()) > 0 else 0
    elif loss_type == 'exp':
        loss = torch.exp(loss)
    elif loss_type == 'expmargin':
        loss = torch.exp(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else 0
    else:
        loss = None
        raise ValueError("loss type must be one of margin, exp and exp_margin, but now is {}".format(loss_type))
    loss = torch.topk(loss, k=tk if tk < loss.size(0) else loss.size(0))[0]
    loss = torch.mean(loss)

    return loss


def calculate_act_loss(act_val=None, loss_type=None, act_margin_val=0.0):
    """
    given the act_val, we return the activate loss to prevent act_val < margin_value ==> act_val - margin_value < 0
    margin_value should be set as a non-negative number
    """
    act_val = (act_margin_val - act_val).squeeze()
    if not loss_type.startswith("top"):
        return loss_func(act_val, loss_type)
    else:
        tk, l_type = loss_type.split("_", 1)
        tk = int(tk[3:])
        return top_k_loss_func(act_val, l_type, tk=tk)
    # return top_k_loss_func(act_val, loss_type)


def calculate_memory_loss(his_act_val=None, act_val=None, loss_type=None, margin_val1=0.0, margin_val2=0.0):
    # margin_val1 should be set as a non-negative number
    # margin_val2 should be set as a non-positive number
    l1_type, l2_type = loss_type.split("+")
    # l1 is to control that act > his_act + margin_value1 ==> act - his_act - margin_val1 > 0
    l1 = (his_act_val - act_val + margin_val1).unsqueeze(dim=1)
    # l2 is to control that margin_val2 > his_act ==> margin_val2 - his_act > 0
    l2 = (his_act_val - margin_val2).squeeze()
    if not l1_type.startswith("top"):
        l1 = loss_func(l1, l1_type)
    else:
        tk, l1_type = l1_type.split("_", 1)
        tk = int(tk[3:])
        # print(l1_type, tk)
        l1 = top_k_loss_func(l1, l1_type, tk=tk)

    if not l2_type.startswith("top"):
        l2 = loss_func(l2, l2_type)
    else:
        tk, l2_type = l2_type.split("_", 1)
        tk = int(tk[3:])
        # print(l1_type, tk)
        l2 = top_k_loss_func(l2, l2_type, tk=tk)

    return l1 + l2


class ModifyLinearOutput(nn.Module):  # nn.Linear(input_size, output_size) -> nn.Linear(input_size, output_size+1)

    def __init__(self,
                 linear: nn.Linear,
                 add_neuron_num=1, init_weight=None, act_loc=0,
                 activate_loss='non_use', memory_loss='non_use',
                 train_memories=None, val_memories=None,
                 drop_num=0, drop_rate=0.5,
                 act_margin_val=0., margin_val1=0., margin_val2=0.,
                 freeze=False,
                 ):
        super().__init__()
        self.linear = copy.deepcopy(linear)
        self.hidden_size = min(self.linear.weight.size())
        self.intermediate_size = max(self.linear.weight.size())
        # self.add_neuron_num is the number of the neuron added
        self.add_neuron_num = add_neuron_num
        # self.add_neuron_loc is the location of the neuron added
        self.add_neuron_loc = [-(i + 1) for i in range(add_neuron_num)]
        # self.act_loc is the location of useful representation in input sequence,
        # such as the 0th hidden states for FC, or all hidden states for zsqa
        self.act_loc = act_loc

        self.extra_output = nn.Linear(self.hidden_size, self.add_neuron_num)
        if init_weight is not None:
            assert init_weight.size(0) == add_neuron_num
            self._reset_parameters(init_weight=init_weight)

        self.activate_loss = activate_loss
        self.memory_loss = memory_loss
        self.act_loss = None
        self.memo_loss = None

        self.train_memory = train_memories
        self.val_memory = val_memories

        self.drop_num = drop_num
        self.drop_rate = drop_rate

        self.act_margin_val = act_margin_val
        self.margin_val1 = margin_val1
        self.margin_val2 = margin_val2

        self.freeze = freeze
        if self.freeze:
            for p in self.extra_output.parameters():
                p.requires_grad = False

    def freeze_self(self):
        for p in self.extra_output.parameters():
            p.requires_grad = False

    def unfreeze_self(self):
        for p in self.extra_output.parameters():
            p.requires_grad = True

    def _reset_parameters(self, init_weight):
        scale = torch.norm(init_weight, dim=-1).unsqueeze(-1)
        self.extra_output.weight = nn.Parameter(init_weight / (scale ** 2))
        init.constant_(self.extra_output.bias, 0.)
        # 初始的激活值为1

    def get_act_loss(self):
        return self.act_loss

    def get_memo_loss(self):
        return self.memo_loss

    def get_dif_dropout(self, h):
        res = [h]
        for i in range(self.drop_num):
            p = self.drop_rate if not isinstance(self.drop_rate, list) else self.drop_rate[i]
            res.append(F.dropout(h, p=p))
        return torch.cat(res, dim=0)

    def forward(self, hidden_states):
        w, b = self.get_modified_weight_bias()
        # if self.drop_num != 0 and self.training:
        #    hidden_states = self.get_dif_dropout(hidden_states)

        output = torch.add(torch.matmul(hidden_states, w.T), b)
        # if not (isinstance(self.act_loc, list) and output.size(1) == 1):
        act_val = None
        if self.activate_loss != 'non_use':
            if isinstance(self.act_loc, list):
                assert len(self.act_loc) == self.add_neuron_num
                act_val = []
                for i, j in enumerate(self.act_loc):
                    try:
                        act_val.append(output[:, j, -self.add_neuron_num + i].view(-1, 1))
                    except:
                        print(f"We met bad case: {j}, {self.add_neuron_num}, {i}")
                        pass
                if len(act_val) > 0:
                    act_val = torch.cat(act_val, dim=1)
                # act_val = output[:, :, self.add_neuron_loc].view(-1, self.add_neuron_num)
            else:
                assert len(self.add_neuron_loc) == 1
                act_val = output[:, self.act_loc, self.add_neuron_loc].view(-1, self.add_neuron_num)
        # act_val_size: [batch_size, add_neuron_num]
        if torch.is_tensor(act_val):
            if self.activate_loss != 'non_use':
                self.act_loss = calculate_act_loss(act_val=act_val, loss_type=self.activate_loss, act_margin_val=self.act_margin_val)
            if self.memory_loss != 'non_use' and not self.memory_loss.startswith('kl'):
                memo = self.train_memory if self.training else self.val_memory
                memo = torch.cat(memo, dim=0) if isinstance(memo, list) else memo
                his_act_val = self.extra_output(memo)
                # (memory_size, add_neuron_num)
                his_act_val = torch.stack([his_act_val for _ in range(act_val.size(0))], dim=0).transpose(0, 1)
                # (memory_size, batch_size, add_neuron_num)
                self.memo_loss = calculate_memory_loss(
                    his_act_val=his_act_val, act_val=act_val, loss_type=self.memory_loss,
                    margin_val1=self.margin_val1, margin_val2=self.margin_val2
                )
        else:
            self.act_loss, self.memo_loss = 0, 0

        return output

    def get_modified_weight_bias(self):
        wd = self.linear.weight.clone().detach()
        we = self.extra_output.weight
        if self.linear.bias is not None:
            bd = self.linear.bias.clone().detach()
        else:
            bd = torch.tensor([0] * wd.size(0)).to(wd.device)
        be = self.extra_output.bias
        # if 0 <= self.loc < self.intermediate_size:
        #     w = torch.cat((wd[:self.loc, ...], we, wd[self.loc + 1:, ...]), dim=0)
        #     b = torch.cat((bd[:self.loc, ...], be, bd[self.loc + 1:, ...]), dim=0)
        # else:
        w = torch.cat((wd, we), dim=0)
        b = torch.cat((bd, be), dim=0)

        return w, b

    def assign_layer(self):
        w, b = self.get_modified_weight_bias()
        new_layer = nn.Linear(self.hidden_size, self.intermediate_size + self.add_neuron_num)
        new_layer.weight = nn.Parameter(w.clone().detach())
        new_layer.bias = nn.Parameter(b.clone().detach())
        new_layer.bias.requires_grad = False
        new_layer.weight.requires_grad = False
        return new_layer


class ModifyLinearInput(nn.Module):  # nn.Linear(input_size, output_size) -> nn.Linear(input_size+1, output_size)
    def __init__(self, linear: nn.Linear, loc: int = -1,
                 amplify=False, freeze_a=False, amplify_con=10.,
                 add_neuron_num=1):
        super().__init__()
        self.linear = copy.deepcopy(linear)
        self.add_neuron_num = add_neuron_num
        self.hidden_size = min(self.linear.weight.size())
        self.intermediate_size = max(self.linear.weight.size())
        self.loc = loc
        self.extra_input = nn.Parameter(torch.randn([self.hidden_size, self.add_neuron_num]))
        self.amplify = amplify
        if self.amplify:
            self.a = nn.Parameter(torch.randn([self.hidden_size, self.add_neuron_num]))
            self.amplify_con = amplify_con
        self._reset_parameters()
        if freeze_a:
            self.a.requires_grad = False

    def freeze_self(self):
        self.extra_input.requires_grad = False
        self.a.requires_grad = False

    def unfreeze_self(self):
        self.extra_input.requires_grad = True
        self.a.requires_grad = True

    def _reset_parameters(self):
        if self.amplify:
            init.constant_(self.a, self.amplify_con)

    def get_modified_weight_bias(self):
        wd = self.linear.weight.clone().detach()
        if self.linear.bias is not None:
            b = self.linear.bias.clone().detach()
        else:
            b = torch.tensor([0] * wd.size(0)).to(wd.device)
        we = self.extra_input
        if self.amplify:
            we = self.extra_input * self.a
        # if 0 <= self.loc < self.intermediate_size:
            # w = torch.cat((wd[..., :self.loc], we, wd[..., self.loc+1:]), dim=1)
        # else:
        w = torch.cat((wd, we), dim=1)
        return w, b

    def forward(self, hidden_states):
        w, b = self.get_modified_weight_bias()
        output = torch.add(torch.matmul(hidden_states, w.T), b)
        return output

    def assign_layer(self):
        w, b = self.get_modified_weight_bias()
        new_layer = nn.Linear(self.intermediate_size + self.add_neuron_num, self.hidden_size)
        new_layer.weight = nn.Parameter(w.clone().detach())
        new_layer.bias = nn.Parameter(b.clone().detach())
        new_layer.bias.requires_grad = False
        new_layer.weight.requires_grad = False

        return new_layer


class ModuleDetector(nn.Module):

    def __init__(self, model: nn.Module, memory_size=None, mode='input', memory_loc=0, hidden_loc=0):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.mode = mode

        self.memory_button = False
        self.hidden_button = False

        self.memory_loc = memory_loc
        self.hidden_loc = hidden_loc

        self.memory = []
        self.tmp_hidden = None

        self.memory_mask = None

    def feed_memory_mask(self, mask):
        self.memory_mask = mask

    def set_memory_loc(self, ml):
        self.memory_loc = ml

    def set_hidden_loc(self, hl):
        self.hidden_loc = hl

    def forward(self, hidden_states):
        output = self.model(hidden_states)

        m_tmp = copy.deepcopy(hidden_states if self.mode == 'input' else output)
        hidden_size = m_tmp.size(-1)
        if self.memory_button:
            if self.memory_loc == 'bart_seq':
                # for seq, we need the seq_mask for masking the padding token
                h = m_tmp[:, 1:, :][self.memory_mask[:, 1:].bool()].reshape(-1, hidden_size)
            else:
                h = m_tmp[:, self.memory_loc].reshape(-1, hidden_size)
            hs = list(torch.split(h, 1))
            self.memory.extend(hs)

        if self.hidden_button:
            if self.hidden_loc == 'bart_seq':
                # The hidden is usually used for one example, we do not need the mask for padding token
                self.tmp_hidden = m_tmp.reshape(-1, hidden_size)
            else:
                self.tmp_hidden = m_tmp[:, self.hidden_loc].reshape(-1, hidden_size)

        return output

    def empty_memory(self):
        self.memory = []

    @staticmethod
    def get_rank_memory(raw_memory, step=40):
        base, tmp_base, memory = [], [], []
        r, prev_r = 0, 0
        bar = tqdm(range(0, len(raw_memory), step))
        for i in bar:
            ms = [m.cpu().detach_().numpy().squeeze() for m in raw_memory[i:i+step]]
            if r < 768:
                tmp_base = ms if len(base) == 0 else base + ms
                r = np.linalg.matrix_rank(np.mat(tmp_base))
            if r > prev_r:
                base = tmp_base
                prev_r = r
                memory.extend(raw_memory[i:i+step])
            else:
                ms = torch.cat(raw_memory[i:i + step], dim=0)
                tmp_memory = torch.cat(memory, dim=0)
                prod = torch.mm(tmp_memory, ms.T)
                mask = torch.sum(prod > 0, dim=0) < len(memory)
                memory.extend(torch.split(ms[mask], 1))
        return memory

    def get_memory(self):
        """
        memory_size = len(self.memory) if memory_size == 'memory_all' else memory_size
        train_m, val_m = list_split(self.memory, ratio=train_len)
        train_m_size = int(len(train_m) if memory_size * train_len > len(train_m) else memory_size * train_len)
        val_m_size = int(len(val_m) if memory_size * (1 - train_len) > len(val_m) else memory_size * (1 - train_len))
        val_memo = random.sample(val_m, val_m_size)
        if method == 'rank':
            train_memo = self.get_rank_memory(raw_memory=train_m)
        else:
            train_memo = random.sample(train_m, train_m_size)
        train_memo = train_memo[:train_m_size] if len(train_memo) > train_m_size else train_memo
        return train_memo, val_memo"""
        return self.memory

    def get_hidden(self):
        return self.tmp_hidden

    def turn_on_memory(self):
        self.memory_button = True

    def turn_off_memory(self):
        self.memory_button = False

    def turn_on_hidden(self):
        self.hidden_button = True

    def turn_off_hidden(self):
        self.hidden_button = False

