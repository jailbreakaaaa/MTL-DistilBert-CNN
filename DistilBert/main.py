from config import Config
from data import read_dataset, prepare_data
from util import setup_seed
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from ranger import Ranger, RangerVA, RangerQH
from train import Train


# def calculate_single_task_score(self, result_dict):
#     """
#     return {"task": [acc, f1]}
#     """
#     df = pd.DataFrame(result_dict)
#     result = {}
#     for task in self.task_ids.keys():
#         y_true, y_pred = df[df["y_label"] == task]["y_true"].to_list(), df[df["y_label"] == task]["y_pred"].to_list()
#         result[task] = {"acc": "%.3f" % accuracy_score(y_true, y_pred),
#                         "f1": "%.3f" % f1_score(y_true, y_pred, average='binary')}
#     result["all"] = {"acc": "%.3f" % accuracy_score(result_dict["y_true"], result_dict["y_pred"]),
#                      "f1": "%.3f" % f1_score(result_dict["y_true"], result_dict["y_pred"], average='binary')}
#     return result

# def main(args):

# frame = inspect.currentframe()
# gpu_tracker = MemTracker(frame)      # 创建显存检测对象
# apparel + baby + books + camera + dvd + electronics + health + imdb + kitchen + magazines + mr + music + software + sports + toys + video
# ["books", "electronics", "dvd", "kitchen_housewares", "apparel", "camera_photo", "health_personal_care", "music", "toys_games", "video", "baby", "magazines", "software", "sports_outdoors", "imdb", "MR"]
# ["books", "electronics", "dvd", "kitchen_housewares", "apparel", "camera_photo", "health_personal_care", "music", "toys_games", "video", "baby", "magazines", "software", "sports_outdoors", "imdb", "MR"
#                   多了四个数据集  , "entertainment", "daily_necessities", "literature", "media"]
# args = Config(task=["books", "electronics", "dvd", "kitchen_housewares", "apparel",
#                     "camera_photo", "health_personal_care", "music", "toys_games", "video",
#                     "baby", "magazines", "software", "sports_outdoors", "imdb", "MR"],
#               model="distilbert_channel", glove_dim=300)
# args = Config(task=["apparel", "camera_photo", "electronics", "kitchen_housewares", "magazines", "sports_outdoors"],
#               model="sd")
# args = Config(task=["apparel", "camera_photo"],
#               model="s")
# teacher_model = TeacherModel(args).to(args.device)
args = Config(task=["health_personal_care"], model="single_cnn", glove_dim=300)
print(args.taskids)
# y有3个值, 句子标签, 任务标签, 句子长度
train_loader, test_loader = prepare_data(args)

# args = Config(task=["apparel", "camera_photo", "electronics"])
if args.model == "s_dis":
    from model.single_channels_dis import generate, Discriminator
    discriminator = Discriminator(sum(args.num_channels), args.task_num).to(args.device)
    optimizer_d = Ranger(discriminator.parameters(), lr=args.lr)
    model = generate(args).to(args.device)
    optimizer = Ranger([{"params": filter(lambda p: p.requires_grad, model.parameters())}],
                       lr=args.lr, weight_decay=args.weight_decay)

    train = Train(args, args.num_epochs, optimizer, optimizer_d)
    train.fit(model, discriminator, train_loader, test_loader)

elif args.model == "s":
    from model.LSTM_attn import generate
elif args.model == "m":
    from model.GRU_attn_Transformer_CNN import generate
elif args.model == "mtc":
    from model.mutil_transformer_cnn import generate
elif args.model == "mfus":
    from model.mutil_channels_fushion import generate
elif args.model == "d":
    from model.GRU_attn2_Transformer_CNN import generate
elif args.model == "sd":
    from model.CNN_GRU_attn_double_emb import generate
elif args.model == "me":
    from model.GRU_attn_Transformer_CNN_emb_fushion import generate
elif args.model == "ec":
    from model.CNN_Transformer import generate
elif args.model == "DP":
    from model.Transformer_GRU_attn_DPCNN import generate
elif args.model == "single_cnn":
    from model.single_task import generate
elif args.model == "single_rnn":
    from model.single_task import generate
elif args.model == "single_transformer":
    from model.single_task import generate
elif args.model == "single_bert":
    from model.single_task import generate
elif args.model == "bert_channel":
    from model.BERT_channels import generate
elif args.model == "distilbert_channel":
    from model.DistilBert_channels import generate
elif args.model == "bert_adv_channel":
    from model.BERT_adv_channels import generate
elif args.model == "no_bert":
    from model.no_BERT_MTL import generate


setup_seed(args.seed)
model = generate(args).to(args.device)
optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
train = Train(args, args.num_epochs, optimizer, None, scheduler)
# for seed in range(5, 50):
#     setup_seed(seed)
train.fit(model, None, train_loader, test_loader)
# train.resume(model, args.resume_path, test_loader, args)



"""
#### 模型2
args = Config(task=["apparel", "camera_photo", "electronics", "kitchen_housewares", "magazines", "sports_outdoors"],
              model="m")
print(args.taskids)
# y有3个值, 句子标签, 任务标签, 句子长度
train_loader, test_loader = prepare_data(args)

# args = Config(task=["apparel", "camera_photo", "electronics"])
if args.model == "s_dis":
    from model.single_channels_dis import generate, Discriminator
    discriminator = Discriminator(sum(args.num_channels), args.task_num).to(args.device)
    optimizer_d = Ranger(discriminator.parameters(), lr=args.lr)
    model = generate(args).to(args.device)
    optimizer = Ranger([{"params": filter(lambda p: p.requires_grad, model.parameters())}],
                       lr=args.lr, weight_decay=args.weight_decay)

    train = Train(args, args.num_epochs, optimizer, optimizer_d)
    train.fit(model, discriminator, train_loader, test_loader)

elif args.model == "s":
    from model.single_channels import generate
elif args.model == "m":
    from model.mutil_channels import generate
elif args.model == "mfus":
    from model.mutil_channels_fushion import generate
setup_seed(args.seed)
model = generate(args).to(args.device)
optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)
train = Train(args, args.num_epochs, optimizer, None, scheduler)
# for seed in range(50):
#     setup_seed(seed)
train.fit(model, None, train_loader, test_loader)


### 模型1
args = Config(task=["apparel", "camera_photo", "electronics", "kitchen_housewares", "magazines", "sports_outdoors"],
              model="ec")
print(args.taskids)
# y有3个值, 句子标签, 任务标签, 句子长度
train_loader, test_loader = prepare_data(args)

# args = Config(task=["apparel", "camera_photo", "electronics"])
if args.model == "s_dis":
    from model.single_channels_dis import generate, Discriminator
    discriminator = Discriminator(sum(args.num_channels), args.task_num).to(args.device)
    optimizer_d = Ranger(discriminator.parameters(), lr=args.lr)
    model = generate(args).to(args.device)
    optimizer = Ranger([{"params": filter(lambda p: p.requires_grad, model.parameters())}],
                       lr=args.lr, weight_decay=args.weight_decay)

    train = Train(args, args.num_epochs, optimizer, optimizer_d)
    train.fit(model, discriminator, train_loader, test_loader)

elif args.model == "s":
    from model.single_channels import generate
elif args.model == "m":
    from model.mutil_channels import generate
elif args.model == "mfus":
    from model.mutil_channels_fushion import generate
elif args.model == "ec":
    from model.encoder_channels import generate
setup_seed(args.seed)
model = generate(args).to(args.device)
optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, step_size=3, gamma=0.95)
train = Train(args, args.num_epochs, optimizer, None, scheduler)
# for seed in range(50):
#     setup_seed(seed)
train.fit(model, None, train_loader, test_loader)


# train.resume(model, args.resume_path, test_loader)

    # model = generate(args).to(args.device)
    # optimizer = Ranger([{"params": filter(lambda p: p.requires_grad, model.parameters())}],
    #                    lr=args.lr, weight_decay=args.weight_decay)
    # train = Train(args, args.num_epochs, optimizer, None)
    # train.fit(model, None, train_loader, test_loader)


# train_dict, test_dict = read_dataset(args)
# print(args.taskids)
# y有3个值, 句子标签, 任务标签, 句子长度
# train_loader, test_loader = prepare_data(args)
# model = generate(args).to(args.device)


# mtl = MultiTaskLossWrapper(model, discriminator, optimizer_d)
"""

"""
当前代码问题：   1.loss的权重设置问题, 需要手动探索loss的值
            2.shared_layer层和private_layer层的参数问题
            3.需要增加discriminator的loss, 在这个基础上研究模型的多共享层问题
            4.保存参数, 把每一个任务的识别率做出来
            5.探究一下test数据为什么长度和train一样
            6.制作多shared层
"""
# loss_params = list(map(id, mtl.log_vars))
# model_params = filter(lambda p: id(p) not in loss_params, mtl.parameters())
# optimizer = Ranger(filter(lambda p: p.requires_grad, mtl.parameters()), lr=args.lr,
#                    weight_decay=args.weight_decay)


"""
旧训练代码
########
train_acc, test_acc = {}, {}
for epoch in range(args.num_epochs):
    pred, true = {"target": [], "task": []}, {"target": [], "task": []}
    for x, label, task_id, seq_len in tqdm(train_loader):

        # label, task_id, seq_len = np.array(list(zip(*y)))

        # target = torch.LongTensor([label, task_id]).to(args.device)

        result, losses = mtl({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])
        # losses, weight = criterion(y_score, label)
        optimizer.zero_grad()
        losses["total"].backward()
        # losses
        optimizer.step()
        pred["target"].extend(result["target"])
        pred["task"].extend(result["task"])
        true["target"].extend(label)
        true["task"].extend(task_id)
    train_acc["target"] = accuracy_score(pred["target"], true["target"])
    train_acc["task"] = accuracy_score(pred["task"], true["task"])
    # print("epoch:{}, test_acc:{}".format(epoch, acc))
    torch.cuda.empty_cache()
    mtl.eval()
    pred, true = {"target": [], "task": []}, {"target": [], "task": []}
    with torch.no_grad():
        for x, label, task_id, seq_len in tqdm(test_loader):

            # pred, true = {"target": [], "task": []}, {"target": [], "task": []}
            result, losses = mtl({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])
            pred["target"].extend(result["target"])
            pred["task"].extend(result["task"])
            true["target"].extend(label)
            true["task"].extend(task_id)
    test_acc["target"] = accuracy_score(pred["target"], true["target"])
    test_acc["task"] = accuracy_score(pred["task"], true["task"])
    print("epoch:{}, train_acc:{}, test_acc:{}".format(epoch, train_acc, test_acc))

    mtl.train()
    torch.cuda.empty_cache()
    # break
print("break")
"""


"""
# 旧代码
teacher_model = TeacherModel(args).to(args.device)
# optimizer = Ranger(filter(lambda p: p.requires_grad, teacher_model.parameters()), lr=args.lr,
#                    weight_decay=args.weight_decay)

student_model = StudentModel(args).to(args.device)
optimizer = Ranger(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr,
                   weight_decay=args.weight_decay)


loss = nn.CrossEntropyLoss()
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# train = Train(args, teacher_model, None, train_loader, test_loader, optimizer, loss, scheduler)
# train.train_teacher()
# train.recurrent(args.best_teacher_model, teacher_model, test_loader)
# train = Train(args, None, student_model, train_loader, test_loader, optimizer, loss, scheduler)
# train.train_student()

train = Train(args, teacher_model, student_model, train_loader, test_loader, optimizer, loss, scheduler, path=args.best_teacher_model)
train.distilling()


# def train():
#
#
#     best_result = {"seed": 0, "epoch": 0, "acc": 0.0, "f1": 0.0}
#
#     for seed in range(args.seed):
#         setup_seed(seed)
#         for epoch in range(args.num_epochs):
#             y_true, y_pred, train_l_sum, batch_count, lr_list = [], [], 0.0, 0.0, []
#             for x, y in tqdm(train_loader):
#                 label, taskid = list(zip(*y))
#                 label = torch.LongTensor(label).to(args.device)
#                 y_score = teachermodel(x)
#                 l = loss(y_score, label)
#
#                 optimizer.zero_grad()
#                 l.backward()
#                 optimizer.step()
#
#                 train_l_sum += l.cpu().item()
#                 batch_count += 1
#                 y_true.extend(label.cpu().tolist())
#                 y_pred.extend(y_score.argmax(dim=1).cpu().tolist())
#             torch.cuda.empty_cache()
#             # gpu_tracker.track()
#             lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
#             scheduler.step()
#
#             print("epoch: {} loss: {:.3f}, lr: {:.5f}, acc: {:.3f}, f1: {:.3f}".format(epoch, train_l_sum/batch_count,
#                                                                            optimizer.state_dict()['param_groups'][0]['lr'],
#                                                                                        accuracy_score(y_true, y_pred),
#                                                                                        f1_score(y_true, y_pred)))
#             best_result = evaluate(teachermodel, best_result, epoch, seed)
#             teachermodel.train()
#
#     print("success")
#
#
# def evaluate(teachermodel, best_result, epoch, seed):
#
#     teachermodel.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for x, y in test_loader:
#             label, taskid = list(zip(*y))
#             label = torch.LongTensor(label).to(args.device)
#             # y = teachermodel.predict(x).cpu()
#             y_true.extend(label.cpu().tolist())
#             y_pred.extend(teachermodel.predict(x).cpu().tolist())
#             torch.cuda.empty_cache()
#             # gpu_tracker.track()
#     result = {"acc": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
#     print("evaluate acc: {:.3f}, f1: {:.3f}".format(result["acc"], result["f1"]))
#     if best_result["acc"] < result["acc"]:
#         best_result["acc"] = result["acc"]
#         best_result["f1"] = result["f1"]
#         best_result["seed"] = seed
#         ## 模型teachermodel
#         if epoch > 1:
#             state = {'model': teachermodel.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
#             path = args.model_path + 'teacher_seed' + str(seed) + '_epoch' + str(epoch + 1) + '_acc' + str(
#                 format(best_result["acc"], '.3f')) + '.pth'
#             torch.save(state, path)
#             print("success save model: ", os.path.abspath(path))
#     return best_result
#
# def load_from_checkpoint(teachermodel, path="./save_checkpoint/teacher_100_3_0.843.pth"):
#     setup_seed(0)
#     checkpoint = torch.load(path)
#     teachermodel.load_state_dict(checkpoint["model"])
#     y_true, y_pred = [], []
#     teachermodel.eval()
#     with torch.no_grad():
#         for x, y in test_loader:
#             label, taskid = list(zip(*y))
#             label = torch.LongTensor(label).to(args.device)
#             # y = teachermodel.predict(x).cpu()
#             y_true.extend(label.cpu().tolist())
#             y_pred.extend(teachermodel.predict(x).cpu().tolist())
#             torch.cuda.empty_cache()
#             # gpu_tracker.track()
#     result = {"acc": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
#     print("evaluate acc: {:.3f}, f1: {:.3f}".format(result["acc"], result["f1"]))

# train()
# load_from_checkpoint(teachermodel)


# if __name__ == '__main__':
# args = Config(task=["apparel", "camera_photo"])
# args = Config(task=["apparel"])
# print(args)
# main(args)
#
"""