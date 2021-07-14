import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
import numpy as np
#
class get_embedding(nn.Module):

    def __init__(self, args):
        super(get_embedding, self).__init__()
        self.args = args
        self.init_glove()
        self.word_dim = args.glove_dim

    def forward(self, x):
        return self.get_glove(x)

    def init_glove(self):
        """
        load the GloVe model
        """
        self.word2id = np.load(self.args.word2id_file, allow_pickle=True).tolist()
        self.glove = nn.Embedding(self.args.vocab_size, self.args.glove_dim)
        emb = torch.from_numpy(np.load(self.args.glove_file, allow_pickle=True)).to(self.args.device)
        self.glove.weight.data.copy_(emb)
        self.word_dim = self.args.glove_dim
        self.glove.weight.requires_grad = False

    def get_glove(self, sentence_lists):
        """
        get the glove word embedding vectors for a sentences
        """
        max_len = max(map(lambda x: len(x), sentence_lists))
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentence_lists))
        sentence_lists = list(map(lambda x: x + [self.args.vocab_size - 1] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists).to(self.args.device)
        # sentence_lists = sentence_lists
        # embeddings = self.glove(sentence_lists)

        return self.glove(sentence_lists)
#
class GRU_attn(nn.Module):
    """
    GRU
    """
    def __init__(self, glove_dim, enc_hid_size, rnn_layers, bidirectional, dec_hid_size, dropout_rnn, device="cuda"):
        super(GRU_attn, self).__init__()
        self.device = device
        # self.args = args
        self.rnn = nn.GRU(glove_dim, enc_hid_size, rnn_layers,
                          batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(enc_hid_size * 2, dec_hid_size)
        else:
            self.fc = nn.Linear(enc_hid_size, dec_hid_size)
        self.attn = Attention(enc_hid_size, dec_hid_size)
        self.dropout = nn.Dropout(dropout_rnn)
        # self.pool = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化
    def forward(self, x, seq_len):
        ## 比长度版本
        # sent_len, idx_sort = np.sort(seq_len)[::-1], np.argsort(-seq_len)
        # idx_unsort = np.argsort(idx_sort)
        # idx_sort = torch.from_numpy(idx_sort).to(self.device)
        # sent_variable = x.index_select(0, idx_sort)
        # # Handling padding in Recurrent Networks
        # sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, np.ascontiguousarray(sent_len, dtype=np.float32),
        #                                                 batch_first=True)
        # sent_output, sent_hidden = self.rnn(sent_packed)
        # sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]        #[batch_size, max_len, 4096]
        # # Un-sort by length
        # idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        # sent_output = sent_output.index_select(0, idx_unsort)  # batch, seq_len, encoder * layer

        ### 普通版本
        sent_output, sent_hidden = self.rnn(x)
        s = torch.tanh(self.fc(torch.cat((sent_hidden[-2, :, :], sent_hidden[-1, :, :]), dim=1)))
        attn_weights = self.attn(s, sent_output)  # batch, seq_len
        # context = attn_weights.bmm(sent_output.transpose(0, 1))
        # local_representation = self.pool(attn_weights.matmul(sent_output)).squeeze(-1)  # batch_size, batch_size, enc_hid_dim
        # batch, enc_hid_size * 2
        local_representation = torch.bmm(sent_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(-1)

        return local_representation

class LSTM_attn(nn.Module):
    """
    GRU
    """
    def __init__(self, glove_dim, enc_hid_size, rnn_layers, bidirectional, dec_hid_size, dropout_rnn, device="cuda"):
        super(LSTM_attn, self).__init__()
        self.device = device
        # self.args = args
        self.rnn = nn.LSTM(glove_dim, enc_hid_size, rnn_layers,
                          batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(enc_hid_size * 2, dec_hid_size)
        else:
            self.fc = nn.Linear(enc_hid_size, dec_hid_size)
        self.attn = Attention(enc_hid_size, dec_hid_size)
        self.dropout = nn.Dropout(dropout_rnn)
        # self.pool = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化
    def forward(self, x):
        ## 比长度版本
        # sent_len, idx_sort = np.sort(seq_len)[::-1], np.argsort(-seq_len)
        # idx_unsort = np.argsort(idx_sort)
        # idx_sort = torch.from_numpy(idx_sort).to(self.device)
        # sent_variable = x.index_select(0, idx_sort)
        # # Handling padding in Recurrent Networks
        # sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, np.ascontiguousarray(sent_len, dtype=np.float32),
        #                                                 batch_first=True)
        # sent_output, sent_hidden = self.rnn(sent_packed)
        # sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]        #[batch_size, max_len, 4096]
        # # Un-sort by length
        # idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        # sent_output = sent_output.index_select(0, idx_unsort)  # batch, seq_len, encoder * layer

        ### 普通版本
        sent_output, (sent_hidden, _) = self.rnn(x)
        s = torch.tanh(self.fc(torch.cat((sent_hidden[-2, :, :], sent_hidden[-1, :, :]), dim=1)))
        attn_weights = self.attn(s, sent_output)  # batch, seq_len
        # context = attn_weights.bmm(sent_output.transpose(0, 1))
        # local_representation = self.pool(attn_weights.matmul(sent_output)).squeeze(-1)  # batch_size, batch_size, enc_hid_dim
        # batch, enc_hid_size * 2
        local_representation = torch.bmm(sent_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(-1)

        return local_representation

class CNN_layers(nn.Module):
    """
    CNN
    """
    def __init__(self, num_channels, kernel_sizes, glove_dim, device="cuda"):
        super(CNN_layers, self).__init__()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=glove_dim,
                                        out_channels=c,
                                        kernel_size=k).to(device))
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight.data)
            nn.init.uniform_(conv.bias, 0, 0)  # 后加入的初始化bias
        self.pool = nn.AdaptiveMaxPool1d(1) #自适应最大池化

    def forward(self, x):
        # x的输出维度为 num_channels * 3
        x = torch.cat([
            self.pool(F.gelu(conv(x))).squeeze(-1) for conv in self.convs], dim=1)

        return x


class TransformerModel(nn.Module):

    # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):  # 需要调整ninp
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)  # ninp是embed_size
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)  # nhid是反馈网络的的尺寸
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.ninp = ninp
        # 暂时不需要解码
        # self.decoder = nn.Linear(ninp, d_model)
        self.pool = nn.AdaptiveMaxPool1d(1)
        # self.init_weights()

    # def generate_square_subsequent_mask(self, sz):  # 生成掩膜向量
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    # def init_weights(self):  # 初始化解码器
    #     initrange = 0.1
    #     # self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src, src_mask):
    def forward(self, src):
        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # output = self.transformer_encoder(src, src_mask)
        output = self.transformer_encoder(src)  # batch, seq_len, embed
        # 可以考虑加入线性层做一个缓冲, GRU_attn含有attention
        # output =
        output = self.pool(output.transpose(1, 2)).squeeze(-1)
        # 暂时不需要解码
        # output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=450):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model是模型的第三维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        # nn.init.
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        # batch_size = enc_output.shape[1]
        src_len = enc_output.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)  # s.shape[batch_size, batch_size, enc_hid_dim]
        # enc_output = enc_output.transpose(0, 1)  # [batch_size, src_len, enc_hid_dim * 2]

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)  # 输出对每一个词的注意力

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, num=2, device="cuda"):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        # self.task_num = task_num
        # loss的权重可以手动调最优
        # self.log_vars = nn.Parameter(torch.ones(num))
        # self.log_vars = nn.Parameter(torch.FloatTensor((0.9, 0.1)))
        # self.total_loss = nn.Parameter(torch.zeros(num))
        self.device = device
        # self.losses = []

    def forward(self, input, targets):
        # losses = []
        outputs = self.model(input)
        targets = torch.LongTensor(targets).to(self.device)
        target_loss = F.cross_entropy(outputs[0], targets[0]) * 0.9
        task_loss = F.cross_entropy(outputs[1], targets[1]) * 0.1

        # dis_loss = F.cross_entropy(outputs[2], targets[1]) * -0.05

        result = {"target": outputs[0].argmax(dim=1).tolist(), "task": outputs[1].argmax(dim=1).tolist()}
        return result, {"target": target_loss, "task": task_loss, "total": target_loss+task_loss}
               # weight.tolist()


class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()
        # self.args = args
        self.embedding = get_embedding(args)
        # self.share_layer = CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)
        # self.share_layer1 = TransformerModel(args.glove_dim, args.nhead, args.nhid, args.nlayers, dropout=args.dropout_trans)
        # 多层GRU
        # self.private_layer = nn.ModuleList([GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
        #                                              args.bidirectional, args.dec_hid_size,
        #                                              args.dropout_rnn)]*args.task_num)  # 多任务对应的decoder
        ## 两层GRU  MT-CNN-GRU
        # self.private_layer0 = GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
        #                                              args.bidirectional, args.dec_hid_size,
        #                                              args.dropout_rnn)
        # self.private_layer1 = GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
        #                                              args.bidirectional, args.dec_hid_size,
        #                                              args.dropout_rnn)
        # 两层CNN  MT-CNN
        # self.private_layer0 = CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)
        # self.private_layer1 = CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)
        # 三层att_lstm  MT-att_LSTM
        self.share_layer = LSTM_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)
        self.private_layer0 = LSTM_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)
        self.private_layer1 = LSTM_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)

        # self.fushion = nn.Linear(sum(args.num_channels)+args.glove_dim, args.enc_hid_size*2)
        if args.bidirectional:
            # CNN+GRU
            # self.fc_target = nn.Linear(sum(args.num_channels)+args.enc_hid_size*2, args.output_size)
            # self.fc_task = nn.Linear(sum(args.num_channels)+args.enc_hid_size*2, args.task_num)
            # all CNN MT-CNN
            # self.fc_target = nn.Linear(sum(args.num_channels)*2, args.output_size)
            # self.fc_task = nn.Linear(sum(args.num_channels)*2, args.task_num)
            # MT-att_LSTM
            self.fc_target = nn.Linear(args.enc_hid_size*4, args.output_size)
            self.fc_task = nn.Linear(args.enc_hid_size*4, args.task_num)

            nn.init.xavier_normal_(self.fc_target.weight)
            nn.init.xavier_normal_(self.fc_task.weight)
            # self.fc_target = nn.Linear(args.enc_hid_size*4, args.output_size)
            # self.fc_task = nn.Linear(args.enc_hid_size*4, args.task_num)
        else:
            self.fc_target = nn.Linear(sum(args.num_channels)+args.enc_hid_size, args.output_size)
            self.fc_task = nn.Linear(sum(args.num_channels)+args.enc_hid_size, args.task_num)
        # self.fc_discriminator = nn.Linear(sum(args.num_channels), args.task_num)

    def forward(self, input):
        # def forward(self, x, task_id, seq_len):
        emb = self.embedding(input["x"])
        # share_layer = self.share_layer(emb.permute(0, 2, 1))  # (batch_size, batch_size)
        # share_layer1 = self.share_layer1(emb)
        # share_layer = self.fushion(torch.cat((share_layer0, share_layer1), dim=1))
        # emb, seq_len
        # 单层GRU
        # private_layer0 = self.private_layer0(emb, input["seq_len"])  # (batch_size, num_channels*3)
        # private_layer1 = self.private_layer1(emb, input["seq_len"])
        # 单层CNN
        # private_layer0 = self.private_layer0(emb.permute(0, 2, 1))  # (batch_size, num_channels*3)
        # private_layer1 = self.private_layer1(emb.permute(0, 2, 1))
        # MT-att_LSTM
        share_layer = self.share_layer(emb)
        private_layer0 = self.private_layer0(emb)  # (batch_size, num_channels*3)
        private_layer1 = self.private_layer1(emb)
        # fusion_layer = torch.cat((share_layer, private_layer), dim=1)
        target = self.fc_target(torch.cat((private_layer0, share_layer), dim=1))
        task = self.fc_task(torch.cat((private_layer1, share_layer), dim=1))

        # discriminator = self.fc_discriminator(share_layer)
        return [target, task, share_layer]
