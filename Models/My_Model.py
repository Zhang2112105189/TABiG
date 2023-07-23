import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from Models.utils import masked_mae_cal
from torch.nn import Softmax


# 前馈层（通过激活函数的方式，来强化表达能力）
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, position_num):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(position_num, hidden_size))

    def _get_sinusoid_encoding_table(self, position_num, hidden_size):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_i // 2) / hidden_size) for hid_i in range(hidden_size)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(position_num)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


# 多头自注意力流程
class MHA(nn.Module):
    def __init__(self, seq_len, in_dim, head_num, attn_dim, inner_dim, dropout=0.1, **kwargs):
        super(MHA, self).__init__()
        self.device = kwargs['device']
        self.seq_len = seq_len
        self.head_num = head_num
        self.attn_dim = attn_dim
        self.layer_norm = nn.LayerNorm(in_dim)
        self.generate_q = nn.Linear(in_dim, head_num * attn_dim)
        self.generate_k = nn.Linear(in_dim, head_num * attn_dim)
        self.generate_v = nn.Linear(in_dim, head_num * attn_dim)
        self.dropout_attn = nn.Dropout(dropout)
        self.softmax = Softmax(dim=-1)
        self.fc = nn.Linear(head_num * attn_dim, in_dim)
        self.pos_ffn = PositionWiseFeedForward(in_dim, inner_dim, dropout)

    def forward(self, x):
        attn_mask = torch.eye(self.seq_len).to(self.device)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        residual = x

        x = self.layer_norm(x)

        batch_size, seq_len, feature_num = x.size()
        Q = self.generate_q(x)
        K = self.generate_k(x)
        V = self.generate_v(x)

        Q = Q.contiguous().view(batch_size, seq_len, self.head_num, self.attn_dim)
        Q = Q.permute(0, 2, 1, 3)

        K = K.contiguous().view(batch_size, seq_len, self.head_num, self.attn_dim)
        K = K.permute(0, 2, 3, 1)

        V = V.contiguous().view(batch_size, seq_len, self.head_num, self.attn_dim)
        V = V.permute(0, 2, 1, 3)

        attn = torch.matmul(Q / self.attn_dim ** 0.5, K)
        attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout_attn(self.softmax(attn))
        output = torch.matmul(attn, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        output = self.fc(output)
        output += residual
        output = self.pos_ffn(output)

        return output, attn


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if self.diag:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, deltas):
        if self.diag:
            gamma = F.relu(F.linear(deltas, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(deltas, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class LSTD(nn.Module):
    def __init__(self, seq_len, feature_num, hidden_size, head_num, attention_size, groups_num, inner_layers_size,
                 dropout_rate, **kwargs):
        super(LSTD, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.device = kwargs['device']

        self.dropout = nn.Dropout(p=dropout_rate)
        self.position_enc = PositionalEncoding(hidden_size, position_num=seq_len)

        self.embedding1 = nn.Linear(feature_num * 3, hidden_size)
        self.MHA_list1 = nn.ModuleList([
            MHA(seq_len, hidden_size, head_num, attention_size, inner_layers_size, dropout_rate, **kwargs)
            for _ in range(groups_num)
        ])
        self.reduce_dim_beta1 = nn.Linear(hidden_size, feature_num)
        self.reduce_dim_gamma1 = nn.Linear(feature_num, feature_num)

        # 用于衰减隐藏状态，目的是在长时间缺失的序列中，减轻很久之前的记录数据对当前的影响
        self.temp_decay_h = TemporalDecay(feature_num, hidden_size, diag=False)
        # 用于参与权重计算，所以采用对角形式
        self.temp_decay_x = TemporalDecay(feature_num, feature_num, diag=True)
        # 用于历史信息回归映射，全连接就是将学到的分布式特征表示映射到样本标记空间，组合特征和分类器功能
        self.hist_reg = nn.Linear(hidden_size, feature_num)
        # self.embedding_comb = nn.Linear(feature_num * 2, feature_num)
        # 用于特征信息回归映射
        self.feat_reg = FeatureRegression(feature_num)  # 对角掩蔽
        # 用于综合考虑 时间间隔 与 掩蔽矩阵 来，计算 历史信息回归结果 与 特征信息回归结果 两者之间的权重
        self.rnn_weight_combine = nn.Linear(feature_num * 2 + seq_len, feature_num)
        # 创建RNN模型所需要的网络对象
        self.gru_cell = nn.GRUCell(feature_num, hidden_size)

        self.ffn = PositionWiseFeedForward(feature_num, inner_layers_size, dropout_rate)

        self.embedding2 = nn.Linear(feature_num * 2, hidden_size)
        self.MHA_list2 = nn.ModuleList([
            MHA(seq_len, hidden_size, head_num, attention_size, inner_layers_size, dropout_rate, **kwargs)
            for _ in range(groups_num)
        ])
        self.reduce_dim_beta2 = nn.Linear(hidden_size, feature_num)
        self.reduce_dim_gamma2 = nn.Linear(feature_num, feature_num)
        self.sa_weight_combine = nn.Linear(feature_num + seq_len, feature_num)

    def forward(self, data, direction):
        values = data[direction]['X'].clone().detach()
        masks = data[direction]['missing_mask'].clone().detach()
        deltas = data[direction]['deltas'].clone().detach()

        Initial_in = torch.cat([values, masks, deltas], dim=2)
        Initial_in = self.embedding1(Initial_in)
        Initial_out = self.dropout(self.position_enc(Initial_in))
        for MHA_layer in self.MHA_list1:
            Initial_out, attn_weights1 = MHA_layer(Initial_out)

        Initial_out = self.reduce_dim_gamma1(F.relu(self.reduce_dim_beta1(Initial_out)))
        attn_weights1 = attn_weights1.squeeze()
        if len(attn_weights1.shape) == 4:
            attn_weights1 = torch.transpose(attn_weights1, 1, 3)
            attn_weights1 = attn_weights1.mean(dim=3)
            attn_weights1 = torch.transpose(attn_weights1, 1, 2)

        gru_in = masks * values + (1 - masks) * Initial_out

        hidden_states = torch.zeros((values.size()[0], self.hidden_size), device=self.device)

        gru_imputed_data = []

        rnn_reconstruction_loss = 0.0

        for t in range(self.seq_len):
            x = gru_in[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            # 通过时间间隔，计算时间衰减量
            gamma_h = self.temp_decay_h(d)
            # 用于后面的权重计算的时间衰减量
            gamma_x = self.temp_decay_x(d)
            # 衰减隐藏状态，目的是在长时间缺失的序列中，减轻很久之前的记录数据对当前的影响
            hidden_states = hidden_states * gamma_h
            # 将从衰减后的隐藏状态中学到的分布式特征表示映射到样本标记空间，即特征维度从hidden size -> feature num
            x_h = self.hist_reg(hidden_states)
            # 使用前面的结果x_h替代特征向量中的空缺值
            # x_r = m * x + (1 - m) * x_h
            # 组合x_c中的特征
            z_h = self.feat_reg(x)
            # 综合考虑时间衰减量与掩蔽矩阵，计算 历史信息回归结果 与 特征信息回归结果 两者之间的权重
            alpha = torch.sigmoid(self.rnn_weight_combine(torch.cat([gamma_x, m, attn_weights1[:, t, :]], dim=1)))
            # 按权重把历史信息回归结果 与 特征信息回归结果结合
            c_h = alpha * z_h + (1 - alpha) * x_h
            # 收集中间结果与插补结果
            gru_imputed_data.append(c_h.unsqueeze(dim=1))
            # 使用c_h替代特征向量中的空缺值
            c_c = m * x + (1 - m) * c_h
            # 连接掩蔽矩阵，即进行特征融合，作为LSTM的输入
            # inputs = torch.cat([c_c, m], dim=1)
            # 输入LSTM，计算下一个隐藏状态
            hidden_states = self.gru_cell(c_c, hidden_states)
            # 计算重建损失
            rnn_reconstruction_loss += masked_mae_cal(x_h, x, m)
            rnn_reconstruction_loss += masked_mae_cal(z_h, x, m)
            rnn_reconstruction_loss += masked_mae_cal(c_h, x, m)

        # 循环结束后，拼接收集的中间结果与插补结果
        gru_imputed_data = torch.cat(gru_imputed_data, dim=1)

        # gru_imputed_data = self.ffn_2(torch.relu(self.ffn_1(gru_imputed_data)))
        gru_imputed_data = self.ffn(gru_imputed_data)

        rnn_reconstruction_loss /= (self.seq_len * 3)
        # 使用插补结果，替代特征向量中的缺失值
        replaced_x = masks * values + (1 - masks) * gru_imputed_data

        input_X = torch.cat([replaced_x, masks], dim=2)
        input_X = self.embedding2(input_X)
        sa_out = self.position_enc(input_X)
        for MHA_layer in self.MHA_list2:
            sa_out, attn_weights2 = MHA_layer(sa_out)

        sa_out = self.reduce_dim_gamma2(torch.relu(self.reduce_dim_beta2(sa_out)))
        attn_weights2 = attn_weights2.squeeze()  # namely term A_hat in math algo
        if len(attn_weights2.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights2 = torch.transpose(attn_weights2, 1, 3)
            attn_weights2 = attn_weights2.mean(dim=3)
            attn_weights2 = torch.transpose(attn_weights2, 1, 2)

        combining_weights = torch.sigmoid(
            self.sa_weight_combine(torch.cat([masks, attn_weights2], dim=2)))  # namely term eta
        # combine X_1 and X_2
        lstd_out = (1 - combining_weights) * sa_out + combining_weights * gru_imputed_data

        reconstruction_loss = 0
        reconstruction_loss += rnn_reconstruction_loss
        reconstruction_loss += masked_mae_cal(Initial_out, values, masks)
        # reconstruction_loss += masked_mae_cal(gru_imputed_data, values, masks)
        reconstruction_loss += masked_mae_cal(lstd_out, values, masks)
        reconstruction_loss /= 3

        rnn_result = {
            'reconstruction_loss': reconstruction_loss,
            'attn_weights': attn_weights2,
            'imputed_data': lstd_out
        }
        return rnn_result


class My_Models(nn.Module):
    def __init__(self, seq_len, feature_num, hidden_size, groups_num, inner_layers_size, head_num, attention_size,
                 dropout_rate, **kwargs):
        super(My_Models, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.groups_num = groups_num
        self.device = kwargs['device']

        self.dropout = nn.Dropout(p=dropout_rate)
        self.position_enc = PositionalEncoding(hidden_size, position_num=seq_len)

        self.f_lstd = LSTD(seq_len, feature_num, hidden_size, head_num, attention_size, groups_num, inner_layers_size,
                           dropout_rate, **kwargs)
        self.b_lstd = LSTD(seq_len, feature_num, hidden_size, head_num, attention_size, groups_num, inner_layers_size,
                           dropout_rate, **kwargs)
        self.fb_weight_combine = nn.Linear(feature_num + seq_len * 2, feature_num)

        self.embedding = nn.Linear(feature_num * 2, hidden_size)
        self.MHA_list = nn.ModuleList([
            MHA(seq_len, hidden_size, head_num, attention_size, inner_layers_size, dropout_rate, **kwargs)
            for _ in range(groups_num)
        ])
        self.reduce_dim_beta = nn.Linear(hidden_size, feature_num)
        self.reduce_dim_gamma = nn.Linear(feature_num, feature_num)
        self.weight_combine = nn.Linear(feature_num + seq_len, feature_num)

    def forward(self, data, stage):
        values = data['forward']['X'].clone().detach()
        masks = data['forward']['missing_mask'].clone().detach()

        f_rnn = self.f_lstd(data, 'forward')
        b_rnn = self.reverse(self.b_lstd(data, 'backward'))

        bidirection_weight = torch.sigmoid(
            self.fb_weight_combine(torch.cat([masks, f_rnn['attn_weights'], b_rnn['attn_weights']], dim=2)))
        bidirection_out = (1 - bidirection_weight) * b_rnn['imputed_data'] + bidirection_weight * f_rnn['imputed_data']

        replaced_x = masks * values + (1 - masks) * bidirection_out

        MHA_in = torch.cat([replaced_x, masks], dim=2)
        MHA_in = self.embedding(MHA_in)
        MHA_out = self.position_enc(MHA_in)
        for MHA_layer in self.MHA_list:
            MHA_out, attn_weights = MHA_layer(MHA_out)

        MHA_out = self.reduce_dim_gamma(torch.relu(self.reduce_dim_beta(MHA_out)))

        attn_weights = attn_weights.squeeze()
        if len(attn_weights.shape) == 4:
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = torch.sigmoid(self.weight_combine(torch.cat([masks, attn_weights], dim=2)))
        final_out = (1 - combining_weights) * MHA_out + combining_weights * bidirection_out

        imputed_data = masks * values + (1 - masks) * final_out

        reconstruction_loss = 0
        reconstruction_loss += f_rnn['reconstruction_loss']
        reconstruction_loss += b_rnn['reconstruction_loss']
        reconstruction_loss += masked_mae_cal(bidirection_out, values, masks)
        # reconstruction_loss += masked_mae_cal(MHA_out, values, masks)
        reconstruction_loss += masked_mae_cal(final_out, values, masks)
        reconstruction_loss /= 4

        consistency_loss = torch.abs(f_rnn['imputed_data'] - b_rnn['imputed_data']).mean() * 1e-1

        reconstruction_MAE = masked_mae_cal(final_out, values, masks)

        if stage != 'test':
            imputation_MAE = masked_mae_cal(final_out, data['X_holdout'], data['indicating_mask'])
        else:
            imputation_MAE = 0.0
        imputation_loss = imputation_MAE

        result = {
            'imputed_data': imputed_data,
            'consistency_loss': consistency_loss,
            'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_loss,
            'reconstruction_MAE': reconstruction_MAE, 'imputation_MAE': imputation_MAE
        }
        return result

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret
