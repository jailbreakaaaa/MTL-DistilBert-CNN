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
        list = [x for x in range(400000)]
        self.word2id = dict(zip(self.word2id,list))
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
        # self.pool = nn.AdaptiveMaxPool1d(1)  # ?????????????????????
    def forward(self, x, seq_len):
        sent_len, idx_sort = np.sort(seq_len)[::-1], np.argsort(-seq_len)
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort).to(self.device)
        sent_variable = x.index_select(0, idx_sort)
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent_variable, np.ascontiguousarray(sent_len, dtype=np.float32),
                                                        batch_first=True)
        sent_output, sent_hidden = self.rnn(sent_packed)
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)[0]        #[batch_size, max_len, 4096]
        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).to(self.device)
        sent_output = self.dropout(F.gelu(sent_output.index_select(0, idx_unsort)))  # batch, seq_len, encoder * layer
        # attention, ?????????dropout,
        s = torch.tanh(self.fc(torch.cat((sent_hidden[-2, :, :], sent_hidden[-1, :, :]), dim=1)))
        attn_weights = self.attn(s, sent_output)  # batch, seq_len
        # context = attn_weights.bmm(sent_output.transpose(0, 1))
        # ?????? # local_representation = self.pool(attn_weights.matmul(sent_output)).squeeze(-1)  # batch_size, batch_size, enc_hid_dim
        # batch, enc_hid_size * 2
        local_representation = torch.bmm(sent_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(-1)
        # ?????????attention
        # local_representation = self.fc(sent_output) ?????????
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
            nn.init.kaiming_uniform_(conv.weight.data)
            nn.init.uniform_(conv.bias, 0, 0)  # ?????????????????????bias
        self.pool = nn.AdaptiveMaxPool1d(1) #?????????????????????

    def forward(self, x):
        # x?????????????????? num_channels * 3
        x = torch.cat([
            self.pool(F.gelu(conv(x))).squeeze(-1) for conv in self.convs], dim=1)

        return x


class TransformerModel(nn.Module):

    # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):  # ????????????ninp
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)  # ninp???embed_size
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)  # nhid???????????????????????????
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.ninp = ninp
        # ?????????????????????
        # self.decoder = nn.Linear(ninp, d_model)
        self.pool = nn.AdaptiveMaxPool1d(1)
        # self.init_weights()

    # def generate_square_subsequent_mask(self, sz):  # ??????????????????
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    # def init_weights(self):  # ??????????????????
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
        # ??????????????????????????????????????????, GRU_attn??????attention
        # output =
        output = self.pool(output.transpose(1, 2)).squeeze(-1)
        # ?????????????????????
        # output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=450):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model????????????????????????
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
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        nn.init.kaiming_uniform_(self.attn.weight.data)
        nn.init.kaiming_uniform_(self.v.weight.data)
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

        return F.softmax(attention, dim=1)  # ?????????????????????????????????

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, num=2, device="cuda"):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        # self.task_num = task_num
        # loss??????????????????????????????
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
        task_loss = F.cross_entropy(outputs[1], targets[0]) * 0.1


        result = {"target": outputs[0].argmax(dim=1).tolist(), "task": outputs[1].argmax(dim=1).tolist()}
        return result, {"target": target_loss, "task": task_loss, "total": target_loss+task_loss}
               # weight.tolist()


class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()
        # self.args = args
        self.embedding = get_embedding(args)
        #self.share_layer0 = CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)
        self.share_layer = TransformerModel(args.glove_dim, args.nhead, args.nhid, args.nlayers, dropout=args.dropout_trans)
        self.private_layer = nn.ModuleList([CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)]*args.task_num)# ??????????????????decoder
        #self.private_layer1 = nn.ModuleList([TransformerModel(args.glove_dim, args.nhead, args.nhid, args.nlayers, dropout=args.dropout_trans)]*args.task_num)
        # self.fushion = nn.Linear(sum(args.num_channels)+args.glove_dim, args.enc_hid_size*2)
        if args.bidirectional:
            # self.fc_target = nn.Linear(sum(args.num_channels)+args.enc_hid_size*2, args.output_size)
            # self.fc_task = nn.Linear(sum(args.num_channels)+args.enc_hid_size*2, args.task_num)
            self.fc_target = nn.Linear(sum(args.num_channels)+args.glove_dim, args.output_size)
            self.fc_task = nn.Linear(sum(args.num_channels)+args.glove_dim, args.task_num)
        else:
            self.fc_target = nn.Linear(sum(args.num_channels)+args.enc_hid_size, args.output_size)
            self.fc_task = nn.Linear(sum(args.num_channels)+args.enc_hid_size, args.task_num)
        # self.fc_discriminator = nn.Linear(sum(args.num_channels), args.task_num)
        self.dropout = nn.Dropout(args.dropout)
    def forward(self, input):
        # def forward(self, x, task_id, seq_len):
        emb = self.embedding(input["x"])
        #share_layer0 = self.share_layer0(emb.permute(0, 2, 1))  # (batch_size, batch_size)
        share_layer = self.share_layer(emb)
        # share_layer = self.fushion(torch.cat((share_layer0, share_layer1), dim=1))
        # emb, seq_len
        private_layer = self.private_layer[input["task_id"]](emb.permute(0, 2, 1))  # (batch_size, num_channels*3)
        # fusion_layer = torch.cat((share_layer, private_layer), dim=1)
        target = self.fc_target(torch.cat((share_layer, private_layer), dim=1))
        task = self.fc_task(torch.cat((share_layer, private_layer), dim=1))

        # discriminator = self.fc_discriminator(share_layer)
        return [target, task, share_layer]
