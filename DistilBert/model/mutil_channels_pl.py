import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from ranger import Ranger, RangerVA, RangerQH
from sklearn.metrics import accuracy_score, f1_score
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
    def __init__(self, args):
        super(GRU_attn, self).__init__()
        self.device = args.device
        # self.args = args
        self.rnn = nn.GRU(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                          batch_first=True, bidirectional=args.bidirectional)
        self.fc = nn.Linear(args.enc_hid_size * 2, args.dec_hid_size)
        self.attn = Attention(args.enc_hid_size, args.dec_hid_size)
        self.dropout = nn.Dropout(args.dropout_rnn)
        self.pool = nn.AdaptiveMaxPool1d(1)  # 自适应最大池化
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
        sent_output = sent_output.index_select(0, idx_unsort)                         #回复原来序列
        # attention, 没有加dropout
        s = torch.tanh(self.fc(torch.cat((sent_hidden[-2, :, :], sent_hidden[-1, :, :]), dim=1)))
        attn_weights = self.attn(s, sent_output)
        # context = attn_weights.bmm(sent_output.transpose(0, 1))
        local_representation = self.pool(attn_weights.matmul(sent_output)).squeeze(-1)  # batch_size, batch_size, enc_hid_dim


        return local_representation

class CNN_layers(nn.Module):
    """
    CNN
    """
    def __init__(self, args):
        super(CNN_layers, self).__init__()
        self.convs = nn.ModuleList()
        for c, k in zip(args.num_channels, args.kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=args.glove_dim,
                                        out_channels=c,
                                        kernel_size=k).to(args.device))
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight.data)
            nn.init.uniform_(conv.bias, 0, 0)  # 后加入的初始化bias
        self.pool = nn.AdaptiveMaxPool1d(1) #自适应最大池化

    def forward(self, x):
        # x的输出维度为 num_channels * 3
        x = torch.cat([
            self.pool(F.gelu(conv(x))).squeeze(-1) for conv in self.convs], dim=1)

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

class MultiTaskLossWrapper(pl.LightningModule):
    def __init__(self, model, args):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.args = args
        # self.task_num = task_num
        # loss的权重可以手动调最优
        # self.log_vars = nn.Parameter(torch.ones(num))
        # self.log_vars = nn.Parameter(torch.FloatTensor((0.9, 0.1)))
        # self.total_loss = nn.Parameter(torch.zeros(num))
        # self.device = device
        # self.losses = []

    def forward(self, input, targets):
        # losses = []
        outputs = self.model(input)
        # targets = torch.LongTensor(targets).to(self.device)
        targets = torch.LongTensor(targets).cuda()
        target_loss = F.cross_entropy(outputs[0], targets[0]) * 0.9
        task_loss = F.cross_entropy(outputs[1], targets[0]) * 0.1

        result = {"target": outputs[0].argmax(dim=1).tolist(), "task": outputs[1].argmax(dim=1).tolist()}
        return result, {"target": target_loss, "task": task_loss, "total": target_loss+task_loss}
               # weight.tolist()

    def configure_optimizers(self):
        optimizer = Ranger([{"params": filter(lambda p: p.requires_grad, self.parameters())}],
                           lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, label, task_id, seq_len = batch
        result, losses = self({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])
        target_acc = accuracy_score(result["target"], label)
        task_acc = accuracy_score(result["task"], task_id)
        return {"loss": losses["total"], "target_acc": target_acc, "task_acc": task_acc}

    def training_epoch_end(self, outputs):
        # tensorboard_logs = outputs[]
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_target_acc = np.mean([x["target_acc"] for x in outputs])
        avg_task_acc = np.mean([x["task_acc"] for x in outputs])
        tensorboard_logs = {'avg_train_loss': avg_loss, 'avg_train_acc': [avg_target_acc, avg_task_acc]}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, label, task_id, seq_len = batch
        result, losses = self({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])
        target_acc = accuracy_score(result["target"], label)
        task_acc = accuracy_score(result["task"], task_id)
        return {"loss": losses["total"], "train_target": target_acc, "task_acc": task_acc}

    def validation_epoch_end(self, outputs):

        return {'val_loss': outputs["loss"], 'progress_bar': outputs}

    def test_step(self, batch, batch_idx):
        x, label, task_id, seq_len = batch
        result, losses = self({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])

        return losses, result


class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()
        # self.args = args
        self.embedding = get_embedding(args)
        self.share_layer = CNN_layers(args)

        self.private_layer = nn.ModuleList([GRU_attn(args)]*args.task_num)  # 多任务对应的decoder
        self.fc_target = nn.Linear(sum(args.num_channels)+args.batch_size, args.output_size)
        self.fc_task = nn.Linear(sum(args.num_channels), args.task_num)
    def forward(self, input):
    # def forward(self, x, task_id, seq_len):
        emb = self.embedding(input["x"])
        share_layer = self.share_layer(emb.permute(0, 2, 1))  # (batch_size, batch_size)
        # emb, seq_len
        private_layer = self.private_layer[input["task_id"]](emb, input["seq_len"])  # (batch_size, num_channels*3)

        target = self.fc_target(torch.cat((private_layer, share_layer), dim=1))
        task = self.fc_task(share_layer)
        return [target, task]
