import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem import WordNetLemmatizer
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np

def collate_fn(batch):
    data, label = zip(*batch)
    # np.array(list(zip(*label)))
    # return data, label
    label, task_id, seq_len = np.array(list(zip(*label)))
    return data, label, task_id, seq_len
# class sampler(Sampler):
#     def __init__(self, data_source):
#         self.data_source = data_source
#
#     def __iter__(self):
#         return iter()
#
#     def __len__(self):
#         return len(self.data_source)

class batch_sampler:
    def __init__(self, sampler, batch_size: int, drop_last: bool, shuff=True) -> None:
        # super(batch_sampler, self).__init__()
        self.sampler = sampler #index_list
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_sum = len(sampler) // batch_size
        self.indexs = list(range(self.batch_sum))
        if shuff:
            np.random.shuffle(self.indexs)

    def __iter__(self):
        for idx in self.indexs:
            low, high = idx * self.batch_size, (idx + 1) * self.batch_size
            yield list(range(low, high))

    def __len__(self):
        return self.batch_sum

class Data(Dataset):
    def __init__(self, x, y):
        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]


def load_data_and_labels(dataset, batch_size, max_len, segmentation=False):
    # max_len = 450
    def shuffle(data):
        x, y = data.values()
        z = list(zip(x, y))
        np.random.shuffle(z)
        return zip(*z)

    x_text, label = [], []
    for id, datas in enumerate(dataset.values()):
        seq_len = []
        x, y = shuffle(datas)
        num = len(y)//batch_size*batch_size
        if segmentation:
            for text in x[:num]:
                # data = data.split()
                if len(text) > max_len:
                    x_text.append(text[:200] + text[-200:])
                    seq_len.append(max_len)
                    # print(len(data[:256] + data[-256:]))
                else:
                    x_text.append(text)
                    seq_len.append(len(text))

            # x_text = [data.split()[:250]+data.split()[-250:] for data in dataset["text"]]
        else:
            # x_text = [data.split() for data in dataset["text"]]
            x_text = x_text[:num]
            seq_len.extend([len(text) for text in x])
        label.extend(list(zip(*[y[:num], [id]*num, seq_len])))

        # y[0].extend(datas["category"])
        # y[1].extend([id]*len(datas["text"]))
    # y = np.array([data for data in dataset["category"]])
    return x_text, label


def prepare_data(args, key=True, workers=0):
    train_data, test_data = read_dataset(args)
    x_train, y_train = load_data_and_labels(train_data, args.batch_size, args.max_len, key) #y是List, 尚未处理为numpy, y[0]是真实标签, y[1]是任务
    x_test, y_test = load_data_and_labels(test_data, args.batch_size, args.max_len, key)
    print("train_data: {}, test_data: {}".format(len(y_train), len(y_test)))
    # a = Data(x_train, y_train[0])
    #
    # x_train, x_test, y_train, y_test = train_test_split(x_text, y, stratify=y, test_size=0.2, random_state=6)
    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop, pin_memory=True, num_workers=workers)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop, pin_memory=True, num_workers=workers)
    # for x, y in test_loader:
    #     print(y)
    #     break
    bs = batch_sampler(list(range(len(x_train))), args.batch_size, args.drop_last_batch, True)
    train_loader = DataLoader(train_data, batch_sampler=bs, collate_fn=collate_fn, pin_memory=True, num_workers=workers)
    # for x, label, task_id, seq_len in train_loader:
    #     print(task_id)
    #     break
    bs = batch_sampler(list(range(len(x_test))), args.batch_size, args.drop_last_batch, False)
    test_loader = DataLoader(test_data, batch_sampler=bs, collate_fn=collate_fn, pin_memory=True, num_workers=workers)

    return train_loader, test_loader

def clean_text(sentence):
    sentence = re.sub("[^a-zA-Z]", " ", sentence).lower()
    wnl = WordNetLemmatizer()
    seq = ""
    for word, tag in pos_tag(sentence.split()):
        if tag.startswith('NN'):
            seq = seq + " " + wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            seq = seq + " " + wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            seq = seq + " " + wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            seq = seq + " " + wnl.lemmatize(word, pos='r')
        else:
            seq = seq + " " + word
    return seq.strip().split()

def read_dataset(args):
    train_dict, test_dict = {}, {}
    def read(path):
        with open(path, "r", encoding='utf-8-sig') as f:
            labels, sentences = [], []
            for line in tqdm(f):
                try:
                    label, sentence = line.strip().split('\t')  # tokens
                    labels.append(int(label))
                    sentences.append(clean_text(sentence))
                except:
                    print("error:", line)
        return {"text": sentences, "category": labels}

    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    for id, task in enumerate(args.task):
        print("get {} dataset".format(task))
        # try:
        train_pickle_path = os.path.join(args.data_path, task + ".train.pickle")
        if os.path.exists(train_pickle_path):
            print("load data from"+train_pickle_path)
            train_dict[task] = load(train_pickle_path)
        else:
            train_dict[task] = read(os.path.join(args.data_path, task + ".task.train"))
            with open(train_pickle_path, 'wb') as handle:
                pickle.dump(train_dict[task], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("success save to "+train_pickle_path)

        test_pickle_path = os.path.join(args.data_path, task + ".test.pickle")
        if os.path.exists(test_pickle_path):
            print("load data from" + test_pickle_path)
            test_dict[task] = load(test_pickle_path)
        else:
            test_dict[task] = read(os.path.join(args.data_path, task + ".task.test"))
            with open(test_pickle_path, 'wb') as handle:
                pickle.dump(test_dict[task], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("success save to "+test_pickle_path)
        args.taskids[task] = id
        args.id_task[id] = task
        # except:
        #     assert ("there is no file")

    return train_dict, test_dict
