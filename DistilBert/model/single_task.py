import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoModel, AutoTokenizer

class get_embedding(nn.Module):
    def __init__(self, args, requires_grad=False):
        super(get_embedding, self).__init__()
        self.args = args
        self.init_glove(requires_grad)
        self.word_dim = args.glove_dim

    def forward(self, x):
        return self.get_glove(x)

    def init_glove(self, requires_grad):
        """
        load the GloVe model
        """
        self.word2id = np.load(self.args.word2id_file, allow_pickle=True).tolist()
        list = [x for x in range(400000)]
        self.word2id = dict(zip(self.word2id, list))
        self.glove = nn.Embedding(self.args.vocab_size, self.args.glove_dim)
        emb = torch.from_numpy(np.load(self.args.glove_file, allow_pickle=True)).to(self.args.device)
        self.glove.weight.data.copy_(emb)
        self.word_dim = self.args.glove_dim
        self.glove.weight.requires_grad = requires_grad

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


class LSTM(nn.Module):
    """
    GRU
    """

    def __init__(self, glove_dim, enc_hid_size, rnn_layers, bidirectional, dec_hid_size, dropout_rnn, device="cuda"):
        super(LSTM, self).__init__()
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
        self.pool = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化

    def forward(self, x):

        sent_output, sent_hidden = self.rnn(x)
        # attention, 没有加dropout
        s = torch.tanh(self.fc(torch.cat((sent_hidden[0][-2, :, :], sent_hidden[0][-1, :, :]), dim=1)))
        attn_weights = self.attn(s, sent_output)  # batch, seq_len

        local_representation = torch.bmm(sent_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(-1)
        return local_representation
        # return self.pool(sent_output.permute(0, 2, 1)).squeeze(-1)



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
            nn.init.uniform_(conv.bias, 0, 0)  # 后加入的初始化bias
        self.pool = nn.AdaptiveMaxPool1d(1) #自适应最大池化

    def forward(self, x):
        # x的输出维度为 num_channels * 3
        x = torch.cat([
            self.pool(F.relu(conv(x))).squeeze(-1) for conv in self.convs], dim=1)
        return x

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
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

class DPCNN(nn.Module):
    def __init__(self, num_filters, hidden_size, num_classes):
        super(DPCNN, self).__init__()
        self.conv_region = nn.Conv2d(1, num_filters, (3, hidden_size), stride=1)
        nn.init.kaiming_uniform_(self.conv_region.weight.data)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        nn.init.kaiming_uniform_(self.conv_region.weight.data)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, word_embs):

        x = word_embs.unsqueeze(1)  # [batch_size, 1, seq_len, embed]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.gelu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.gelu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 1:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut
        return x

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
        self.decoder = nn.Linear(ninp, 64)
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
        output = self.decoder(output)
        output = self.pool(output.transpose(1, 2)).squeeze(-1)
        # 暂时不需要解码
        # output = self.decoder(output)
        return output

class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()
        # self.args = args
        # 单任务
        self.embedding = get_embedding(args)
        # self.fc_emb = nn.Linear(args.glove_dim, args.min_emb)
        # nn.init.xavier_uniform(self.fc_emb.weight)
        # args.glove_dim = args.min_emb
        # CNN
        # self.classification = CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)
        # LSTM
        self.classification = LSTM(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     True, args.dec_hid_size,
                                                     args.dropout_rnn)
        # transformer encoder
        self.classification = TransformerModel(args.glove_dim, args.nhead, args.nhid, args.nlayers, dropout=args.dropout_trans)
        # self.embedding = BERT(args)
        # self.share_layer1 = TransformerModel(args.glove_dim, args.nhead, args.nhid, args.nlayers, dropout=args.dropout_trans)
        # self.share_layer2 = DPCNN(args.num_filters, args.hidden_size, args.out_size)
        # self.private_layer = nn.ModuleList([GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
        #                                              args.bidirectional, args.dec_hid_size,
        #                                              args.dropout_rnn)]*args.task_num)  # 多任务对应的decoder
        # self.private_layer = nn.ModuleList([TransformerModel(args.glove_dim, args.nhead, args.nhid, args.nlayers, dropout=args.dropout_trans)]*args.task_num)
        # self.fc = nn.Linear(sum(args.num_channels)+args.glove_dim, args.enc_hid_size*2)
        if args.bidirectional:
            # CNN
            # self.fc_target = nn.Linear(sum(args.num_channels), args.output_size)
            # LSTM
            # self.fc_target = nn.Linear(args.enc_hid_size*2, args.output_size)
            # transformer encoder
            self.fc_target = nn.Linear(64, args.output_size)
            # self.fc_task = nn.Linear(512, args.task_num)
            nn.init.xavier_uniform_(self.fc_target.weight)
            # nn.init.xavier_uniform_(self.fc_task.weight)
            # self.fc_target = nn.Linear(args.enc_hid_size*4, args.output_size)
            # self.fc_task = nn.Linear(args.enc_hid_size*4, args.task_num)
        else:
            self.fc_target = nn.Linear(512, args.output_size)
            self.fc_task = nn.Linear(512, args.task_num)
        # self.fc_discriminator = nn.Linear(sum(args.num_channels), args.task_num)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        # def forward(self, x, task_id, seq_len):
        emb = self.embedding(input["x"])

        # emb = F.gelu(self.fc_emb(emb))
        # CNN
        # classification = self.dropout(self.classification(emb.permute(0, 2, 1)))  # (batch_size, batch_size)
        # LSTM
        classification = self.dropout(self.classification(emb))
        # share_layer1 = self.share_layer1(emb)
        # share_layer = self.fushion(torch.cat((share_layer0, share_layer1), dim=1))
        # emb, seq_len
        # private_layer = self.dropout(self.private_layer[input["task_id"]](emb, input["seq_len"]))  # (batch_size, num_channels*3)
        # private_layer = self.dropout(self.private_layer[input["task_id"]](emb))
        # fusion_layer = torch.cat((share_layer, private_layer), dim=1)
        target = self.fc_target(classification)
        # task = self.fc_task(emb)

        # discriminator = self.fc_discriminator(share_layer)
        return [target, None, None]

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args
        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        self.bert = AutoModel.from_pretrained(args.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.word_dim = args.bert_dim
        self.dropout = nn.Dropout(args.dropout)  # 丢弃层用于防止过拟合
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.encoder = nn.Linear(args.bert_dim, 512)
        nn.init.xavier_uniform_(self.encoder.weight.data)
        self.decoder = nn.Linear(512, 2)
        nn.init.xavier_uniform_(self.decoder.weight.data)

    def forward(self, x):

        word_emb = self.get_embedding(x)
        x = self.encoder(word_emb).permute(0, 2, 1)
        x = self.pool(F.relu(x)).squeeze(-1)
        # x = self.decoder(self.dropout(x))

        return x

    def get_embedding(self, sentence_lists):
        sentence_lists = [' '.join(x) for x in sentence_lists]
        # print("seqlen:", len(sentence_lists[0]), len(sentence_lists[1]))
        ids = self.bert_tokenizer(sentence_lists, padding=True, return_tensors="pt")
        inputs = ids['input_ids']
        if self.args.use_gpu:
            inputs = inputs.to(self.args.device)
        return self.bert(inputs)[0]
