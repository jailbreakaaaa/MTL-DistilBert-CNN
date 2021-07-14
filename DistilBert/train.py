import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from util import setup_seed
import os
# import pytorch_lightning as pl
# pl.Trainer
# np.set_printoptions(precision=3)

class Train(object):

    def __init__(self, args, epochs, optimizer, optimizer_d, scheduler):
        self.args = args
        self.epoch = epochs
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.scheduler = scheduler

        # self.loss = {}
        # self.train_acc = {"target": {"all": {"acc": 0, "f1": 0}}, "task": {"all": {"acc": 0, "f1": 0}}}
        # self.best_acc = {"target": {"acc": 0, "f1": 0}, "task": {"acc": 0, "f1": 0}}
        # self.test_acc = {"target_acc": {"all": 0}, "target_f1": {"all": 0}, "task_acc": {"all": 0}, "task_f1": {"all": 0}}

        self.train_acc = {"target": {"all": {"acc": 0, "f1": 0}}}
        self.best_acc = {"target": {"acc": 0, "f1": 0}}
        self.test_acc = {"target_acc": {"all": 0}, "target_f1": {"all": 0}}

        self.result_path = args.result_path

    def fit(self, model, discriminator, train_loader, test_loader):
        # setup_seed(self.args.seed)
        for epoch in range(self.epoch):
            loss = self.train_epoch(model, discriminator, train_loader)

            self.test_epoch(model, test_loader)
            if self.best_acc["target"]["acc"] < self.test_acc["target_acc"]["all"] and epoch > 0:
                self.save_checkpoint(model, epoch)

            print("epoch: {} loss: {}\n train_acc: {}\n test_acc:{}".format(epoch, loss, self.train_acc, self.test_acc))
            # print()

    def train_epoch(self, model, discriminator, train_loader):
        pred, true, losses = {"target": [], "task": []}, {"target": [], "task": []}, \
                           {"target": [], "task": [], "dis_loss": [], "dis_loss0": [], "total": []}
        for x, label, task_id, seq_len in tqdm(train_loader):
            # label, task_id, seq_len = np.array(list(zip(*y)))

            targets = torch.LongTensor([label, task_id]).cuda()

            # result, loss = model({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])


            outputs = model({"x": x, "task_id": task_id[0], "seq_len": seq_len})
            # targets = torch.LongTensor(targets).to(self.device)
            target_loss = F.cross_entropy(outputs[0], targets[0]) * self.args.target_weight
            if not outputs[1]==None:  # 判断是否为N
                task_loss = F.cross_entropy(outputs[1], targets[1]) * self.args.task_weight
                total_loss = target_loss + task_loss
            else:
                total_loss = target_loss

            # result = {"target": outputs[0].argmax(dim=1).tolist(), "task": outputs[1].argmax(dim=1).tolist()}
            # loss = {"target": target_loss, "task": task_loss, "dis_loss": dis_loss, "dis_loss0": dis_loss0,
            #                 "total": target_loss + task_loss + dis_loss}
            # loss = {"target": target_loss, "task": task_loss, "total": target_loss + task_loss}
            # if

            self.optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.args.max_norm)
            self.optimizer.step()
            self.scheduler.step()

            losses["target"].append(target_loss.tolist())
            #losses["task"].append(task_loss.tolist())

            # losses["dis_loss"].append(loss["dis_loss"].tolist())
            # losses["dis_loss0"].append((loss["dis_loss0"].tolist()))
            losses["total"].append(total_loss.tolist())

            pred["target"].extend(outputs[0].argmax(dim=1).tolist())
            #pred["task"].extend(outputs[1].argmax(dim=1).tolist())
            true["target"].extend(label)
            #true["task"].extend(task_id)

        # self.train_acc["target"]["acc"] = accuracy_score(pred["target"], true["target"])
        # self.train_acc["target"]["f1"] = f1_score(pred["target"], true["target"])
        # self.train_acc["task"]["acc"] = accuracy_score(pred["task"], true["task"])
        # self.train_acc["task"]["f1"] = f1_score(pred["task"], true["task"], average="macro")
        self.train_acc["target"] = accuracy_score(pred["target"], true["target"])
        #self.train_acc["task"] = accuracy_score(pred["task"], true["task"])
        # print("epoch:{}, test_acc:{}".format(epoch, acc))
        torch.cuda.empty_cache()


        # return {"target": np.mean(losses["target"]), "task": np.mean(losses["task"]), "dis_loss": np.mean(losses["dis_loss"]), "total": np.mean(losses["total"])}
        return {"target": np.mean(losses["target"]), "total": np.mean(losses["total"])}

    def test_epoch(self, model, test_loader):
        model.eval()
        # pred, true = {"target": [], "task": []}, {"target": [], "task": []}
        pred, true, results = {"target": {}, "task": {}}, {"target": {}, "task": {}}, {"target": {"all": {}}, "task": {"all": {}}}
        # all_acc, all_f1 = {"target": [], "task": []}, {"target": [], "task": []}
        for task in self.args.task:
            pred["target"][task] = []
            # pred["task"][task] = []
            true["target"][task] = []
            # true["task"][task] = []
        with torch.no_grad():
            for x, label, task_id, seq_len in tqdm(test_loader):

                # pred, true = {"target": [], "task": []}, {"target": [], "task": []}
                # result, losses = model({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])
                result = model({"x": x, "task_id": task_id[0], "seq_len": seq_len})
                pred["target"][self.args.id_task[task_id[0]]].extend(result[0].argmax(dim=1).tolist())
                # pred["task"][self.args.id_task[task_id[0]]].extend(result[1].argmax(dim=1).tolist())
                true["target"][self.args.id_task[task_id[0]]].extend(label)
                # true["task"][self.args.id_task[task_id[0]]].extend(task_id)
        self.test_acc = self.evaluate(true, pred, self.args.task)
        # self.test_acc["target"]["acc"] = accuracy_score(pred["target"], true["target"])
        # self.test_acc["target"]["f1"] = f1_score(pred["target"], true["target"])
        # self.test_acc["task"]["acc"] = accuracy_score(pred["task"], true["task"])
        # self.test_acc["task"]["f1"] = f1_score(pred["task"], true["task"], average="macro")

        # print("epoch:{}, train_acc:{}, test_acc:{}".format(epoch, train_acc, test_acc))

        model.train()
        torch.cuda.empty_cache()

    def save_checkpoint(self, model, epoch):
        self.best_acc["target"]["acc"] = self.test_acc["target_acc"]["all"]
        # self.best_acc["target"]["f1"] = self.test_acc["target"]["all"]["f1"]
        # self.best_acc["task"]["acc"] = self.test_acc["task"]["all"]["acc"]
        # self.best_acc["task"]["f1"] = self.test_acc["task"]["all"]["f1"]
        state = {'model': model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch + 1}
        model_name = self.args.model_path + self.args.model + '_' + str(self.args.seed) +\
                     '_' + str(epoch + 1) + '_' + str(format(self.best_acc["target"]["acc"], '.3f')) + '.pth'
        print("model save in {}".format(model_name))
        torch.save(state, model_name)
        result_path = self.result_path + str(format(self.best_acc["target"]["acc"], '.3f')) + "_" + self.args.model_path.split("/")[-2] + ".csv"
        pd.DataFrame(self.test_acc).to_csv(result_path)

    def evaluate(self, true, pred, tasks):
        results = {"target_acc": {"all": 0}, "target_f1": {"all": 0}, "task_acc": {"all": 0}, "task_f1": {"all": 0}}
        for task in tasks:
            results["target_acc"][task] = 0
            results["target_f1"][task] = 0
            # results["task_acc"][task] = 0
            # results["task_f1"][task] = 0

        all_acc, all_f1 = {"target": [], "task": []}, {"target": [], "task": []}
        for task, true_value, pred_value in zip(true["target"].keys(), true["target"].values(),
                                                pred["target"].values()):
            results["target_acc"][task] = accuracy_score(true_value, pred_value).round(3)
            results["target_f1"][task] = f1_score(true_value, pred_value).round(3)
            all_acc["target"].append(results["target_acc"][task])
            all_f1["target"].append(results["target_f1"][task])
        # for task, true_value, pred_value in zip(true["task"].keys(), true["task"].values(),
        #                                         pred["task"].values()):
        #     results["task_acc"][task] = accuracy_score(true_value, pred_value).round(3)
        #     results["task_f1"][task] = f1_score(true_value, pred_value, average="macro").round(3)
        #     all_acc["task"].append(results["task_acc"][task])
        #     all_f1["task"].append(results["task_f1"][task])
        results["target_acc"]["all"] = np.mean(all_acc["target"]).round(3)
        results["target_f1"]["all"] = np.mean(all_f1["target"]).round(3)
        # results["task_acc"]["all"] = np.mean(all_acc["task"]).round(3)
        # results["task_f1"]["all"] = np.mean(all_f1["task"]).round(3)

        # print(results["target"], "\n", results["task"])
        return results


    def resume(self, model, model_path, test_loader, args):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        pred, true, results = {"target": {}, "task": {}}, {"target": {}, "task": {}}, {"target": {"all": {}}, "task": {"all": {}}}
        # all_acc, all_f1 = {"target": [], "task": []}, {"target": [], "task": []}
        for task in args.task:
            pred["target"][task] = []
            # pred["task"][task] = []
            true["target"][task] = []
            # true["task"][task] = []
            # results["target"][task] = {}
            # results["task"][task] = {}

        # self.test_epoch(model, test_loader)
        # print(self.test_acc)
        model.eval()
        # pred, true = {"target": [[]*task_num], "task": [[]*task_num]}, {"target": [[]*task_num], "task": [[]*task_num]}
        with torch.no_grad():
            for x, label, task_id, seq_len in tqdm(test_loader):
                # pred, true = {"target": [], "task": []}, {"target": [], "task": []}
                # result, losses = model({"x": x, "task_id": task_id[0], "seq_len": seq_len}, [label, task_id])
                result = model({"x": x, "task_id": task_id[0], "seq_len": seq_len})
                pred["target"][args.id_task[task_id[0]]].extend(result[0].argmax(dim=1).tolist())
                # pred["task"][args.id_task[task_id[0]]].extend(result[1].argmax(dim=1).tolist())
                true["target"][args.id_task[task_id[0]]].extend(label)
                # true["task"][args.id_task[task_id[0]]].extend(task_id)
                # pred["target"].extend(result[0].argmax(dim=1).tolist())
                # pred["task"].extend(result[1].argmax(dim=1).tolist())
                # pred["target"].extend(result["target"])
                # pred["task"].extend(result["task"])
                # true["target"].extend(label)
                # true["task"].extend(task_id)
        results = self.evaluate(true, pred, args.task)
        # for task, true_value, pred_value in zip(true["target"].keys(), true["target"].values(), pred["target"].values()):
        #     results["target"][task]["acc"] = accuracy_score(true_value, pred_value).round(3)
        #     results["target"][task]["f1"] = f1_score(true_value, pred_value).round(3)
        #     results["task"][task]["acc"] = accuracy_score(true_value, pred_value).round(3)
        #     results["task"][task]["f1"] = f1_score(true_value, pred_value).round(3)
        #     all_acc["target"].append(results["target"][task]["acc"])
        #     all_acc["task"].append(results["task"][task]["acc"])
        #     all_f1["target"].append(results["target"][task]["f1"])
        #     all_f1["task"].append(results["task"][task]["f1"])
        # results["target"]["all"]["acc"] = np.mean(all_acc["target"]).round(3)
        # results["target"]["all"]["f1"] = np.mean(all_f1["target"]).round(3)
        # results["task"]["all"]["acc"] = np.mean(all_acc["task"]).round(3)
        # results["task"]["all"]["f1"] = np.mean(all_f1["task"]).round(3)

        print(results)



        # self.test_acc["target"]["acc"] = accuracy_score(true["target"], pred["target"])
        # self.test_acc["target"]["f1"] = f1_score(true["target"], pred["target"])
        # self.test_acc["task"]["acc"] = accuracy_score(true["task"], pred["task"])
        # self.test_acc["task"]["f1"] = f1_score(true["task"], pred["task"], average="macro")