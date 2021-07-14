class Config():
    """
    Namespace(adversarial=True, average_loss=False, batch_size=10, beta=0.05, bidirection=False, cuda=True,
    data='../data/', dropout=0.2, dropout_fc=0, emsize=256, emtraining=False, epochs=100, fc_dim=512, gamma=0.01,
    lr_decay=0.99, lrshrink=5, max_example=-1, max_norm=5.0, minlr=1e-05, nhid=1024, nlayers=1, nonlinear_fc=False,
    optfile='./models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json', optimizer='sgd,lr=0.1', plot_every=2000,
    pool_type='mean', print_every=20, resume='', rnn_type='LSTM', save_path='adversarial_output/', seed=1111, start_epoch=0,
    task='apparel + baby + books + camera + dvd + electronics + health + imdb + kitchen + magazines + mr + music + software + sports + toys + video',
    tokenize=False, use_elmo=False, wgtfile='./models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
    word_vectors_directory='../data/', word_vectors_file='embedding.txt')
    """

    def __init__(self, task, model, glove_dim=300):
        self.data_path = "../data/"  # args.data
        self.model = model
        self.seed = 99
        self.task = task
        self.task_num = len(task)
        self.bert_path = "../data/bert/"
        self.distilbert_path = "../data/distilbert/"
        self.bert_dim = 768
        self.distilbert_dim = 768
        self.segmentation = True
        # self.batch_size = 320
        self.max_len = 420  # 最大似乎是450
        self.batch_size = 1   # multi-task: 100; no_bert: 400
        self.taskids = {}
        self.id_task = {}
        self.drop_last_batch = False
        self.workers = 0  # windows设置为0
        self.device = "cuda"
        self.use_gpu = True
        self.num_channels = [64, 64, 64]  # 80 96 # CNN
        # self.single_channel = 72
        self.kernel_sizes = [3, 4, 5]  # CNN
        self.num_filters = 2  # DPCNN
        self.hidden_size = 300  # DPCNN
        self.out_size = 32
        self.pool_type = "max"
        self.lr = 0.2
        self.num_epochs = 10
        self.dropout = 0.5
        self.dropout_rnn = 0.5
        self.dropout_trans = 0.5

        if model == "s":
            self.model_path = "./save_checkpoint/LSTM_attn/"
            self.resume_path = "./save_checkpoint/LSTM_attn/s_100_8_0.887.pth"
        elif model == "m":
            self.model_path = "./save_checkpoint/GRU_attn_Transformer_CNN/"
            self.resume_path = "./save_checkpoint/GRU_attn_Transformer_CNN/"
        elif model == "mtc":
            self.model_path = "./save_checkpoint/mutil_transformer_cnn/"
            self.resume_path = "./save_checkpoint/mutil_transformer_cnn/"
        elif model == "mfus":
            self.model_path = "./save_checkpoint/mutil_fushion_channel/"
            self.resume_path = "./save_checkpoint/mutil_fushion_channel/mfus_100_6_0.870.pth"
        elif model == "d":
            self.model_path = "./save_checkpoint/GRU_attn2_Transformer_CNN/"
            self.resume_path = "./save_checkpoint/GRU_attn2_Transformer_CNN/d_100_11_0.874.pth"
        elif model == "sd":
            self.model_path = "./save_checkpoint/CNN_GRU_attn_double_emb/"
            self.resume_path = None
        elif model == "me":
            self.model_path = "./save_checkpoint/GRU_attn_Transformer_CNN_emb_fushion/"
            self.resume_path = None
        elif model == "ec":
            self.model_path = "./save_checkpoint/CNN_Transformer /"
            self.resume_path = None
        elif model == "DP":
            self.model_path = "./save_checkpoint/Transformer_GRU_attn_DPCNN/"
            self.resume_path = None

        elif model == "single_cnn":
            self.model_path = "./save_checkpoint/single/cnn"
            self.resume_path = None
        elif model == "single_rnn":
            self.model_path = "./save_checkpoint/single/rnn"
            self.resume_path = None
        elif model == "single_birnn":
            self.model_path = "./save_checkpoint/single/birnn"
            self.resume_path = None
        elif model == "single_dpcnn":
            self.model_path = "./save_checkpoint/single/dpcnn"
            self.resume_path = None
        elif model == "single_transformer":
            self.model_path = "./save_checkpoint/single/transformer"
            self.resume_path = None
        elif model == "single_bert":
            self.model_path = "./save_checkpoint/single/bert"
            self.resume_path = None
        elif model == "bert_channel":
            self.model_path = "./save_checkpoint/bert_channel/"
            self.resume_path = None
        elif model == "bert_adv_channel":
            self.model_path = "./save_checkpoint/bert_adv_channel/"
            self.resume_path = None
        elif model == "no_bert":
            self.model_path = "./save_checkpoint/no_bert"
            self.resume_path = None
        elif model == "distilbert_channel":
            self.model_path = "./save_checkpoint/distilbert_channel/"
            self.resume_path = None

        self.result_path = "./result/"

        if glove_dim == 300:
            self.vocab_size = 400000
            self.glove_dim = 300
            self.glove_file = "../data/glove/glove_300d.npy"
            self.word2id_file = "../data/glove/word2id.npy"
        elif glove_dim == 200:
            self.vocab_size = 400000
            self.glove_dim = 200
            self.glove_file = "../data/glove/glove_200d.npy"
            self.word2id_file = "../data/glove/word2id_40.npy"
        elif glove_dim == 100:
            self.vocab_size = 400000
            self.glove_dim = 100
            self.glove_file = "../data/glove/glove_100d.npy"
            self.word2id_file = "../data/glove/word2id_40.npy"
        elif glove_dim == 50:
            self.vocab_size = 400000
            self.glove_dim = 50
            self.glove_file = "../data/glove/glove_50d.npy"
            self.word2id_file = "../data/glove/word2id_40.npy"

        self.min_emb = 100
        self.num_heads = 4
        self.weight_decay = 0.1
        self.max_norm = 0.9

        self.ndim = 256
        self.nhead = 4  # multi-task 2  # Transformer
        self.nhid = 64  # multi-task 32  # Transformer
        self.nlayers = 6  # multi-task 2  # Transformer

        self.target_weight = 1
        self.task_weight = 0.05
        self.diff_weight = 0.01

        # self.num_filters = 512
        self.enc_hid_size = 64
        self.dec_hid_size = 64
        self.num_classes = 2
        self.num_directions = 1
        # self.num_layers = 128
        self.rnn_layers = 1
        self.output_size = 2  # 模型最后的输出大小, 2分类问题

        # self.best_teacher_model = "./best_checkpoint/teacher_seed4_epoch14_acc0.991.pth"
        self.alpha = 1
        self.bidirectional = True
        # self.task = ["apparel", "camera_photo", "electronics", "kitchen_housewares", "magazines",
        #              "sports_outdoors"]  # 设置任务数量

        # self.optfile = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
        # self.wgtfile = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


# args = Config()
# print(args.task)
