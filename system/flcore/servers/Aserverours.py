import copy
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from system.flcore.clients.Aclientours import clientOurs
from system.flcore.servers.serverbase import Server

pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 设置显示宽度，None表示不限制宽度

from torch.autograd import Function


class OursFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientOurs)
        self.personalized_models = [copy.deepcopy(args.model) for _ in range(self.num_clients)]
        # self.updated_deltaW = [[copy.deepcopy(args.model) for _ in range(self.num_clients)] for _ in
        #                        range(self.num_clients)]
        self.uploaded_list = [[] for _ in range(self.num_clients)]
        self.personalized_list = [[] for _ in range(self.num_clients)]
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.asymmetric_relation = np.array([[1.0 for _ in range(self.num_clients)] for _ in range(self.num_clients)])
        for i in range(self.num_clients):
            self.asymmetric_relation[i][i] = 1.0
        # self.load_model()
        self.global_lr = args.global_lr
        self.ar_lr = args.ar_lr
        self.Budget = []
        self.gradients = [[] for _ in range(self.num_clients)]

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            print("\nTrain personalized models")
            #############
            self.selected_clients = self.select_clients()
            self.personalized_transit_point()
            #############
            c_t = time.time()
            for client in self.selected_clients:
                client.train()
                self.gradients[client.id] = client.collect_gradients(self.personalized_models[client.id])
                # print(self.gradients[client.id])
                self.gradients[client.id].clamp_(min=-5.0, max=5.0)
            c_t = time.time() - c_t
            print('-' * 25, 'client time', '-' * 25, c_t)
            #############
            self.receive_models()
            self.get_listW()
            # self.get_deltaW()
            #############
            with torch.no_grad():
                if i != 0:
                    trainable_asymmetric_relation = torch.tensor(self.asymmetric_relation, device=self.device)
                    for client in self.selected_clients:
                        for idx in self.uploaded_ids:
                            if trainable_asymmetric_relation[client.id][idx] == 0:
                                continue
                            deltaW = self.uploaded_list[idx] - self.personalized_list[client.id]
                            deltaW = deltaW / (deltaW.norm(p=2) + 1e-12)
                            gradients_k_idx = self.global_lr * self.gradients[client.id].T @ deltaW
                            trainable_asymmetric_relation[client.id][idx] = min(max(
                                trainable_asymmetric_relation[client.id][idx] - self.ar_lr * gradients_k_idx, 0.0), 5)
                    self.asymmetric_relation = trainable_asymmetric_relation.cpu().detach().numpy()
            #############
            self.personalized_aggregate()
            #############
            for k in range(self.num_clients):
                self.clients[k].set_parameters(self.personalized_models[k])
            #############
            if i % 10 == 0:
                print("\nGain Relationship Matrix")
                df = pd.DataFrame(self.asymmetric_relation)

                pd.options.display.float_format = '{:.3f}'.format
                print(df)
            #     # # 绘制热力图
            #     plt.imshow(np.log(df + 1) / np.log(6), cmap="turbo", vmin=0, vmax=1)  # 设置颜色映射范围
            #     plt.colorbar()  # 添加颜色条
            #     plt.title("Contribution Relation Matrix")  # 添加标题
            #     plt.xticks(np.arange(self.num_clients), np.arange(self.num_clients))  # X轴刻度范围为 0-19
            #     plt.yticks(np.arange(self.num_clients), np.arange(self.num_clients))  # Y轴刻度范围为 0-19
            #     plt.show()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("Evaluate personalized models")
                self.evaluate()
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print("\nBest accuracy.")

        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        print(self.rs_test_acc)
        print(self.rs_train_loss)
        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientOurs)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def params_tolist(self, model):
        model.eval()
        params_list = []
        for param in model.parameters():
            params_list.append(param.view(-1))
        params_tensor = torch.cat(params_list).to(self.device)
        return params_tensor

    def list_load_to_params(self, model, params_tensor):
        params = list(model.parameters())
        split_sizes = [p.numel() for p in params]
        split_tensors = torch.split(params_tensor, split_sizes)

        with torch.no_grad():
            for p, split_tensor in zip(params, split_tensors):
                new_param = split_tensor.view(p.shape)
                p.data.copy_(new_param)

    def personalized_transit_point(self):
        for selected_client in self.selected_clients:
            for param in selected_client.agent_model.parameters():
                param.data.zero_()
            self.personalized_weights = []
            for client in self.selected_clients:
                k = selected_client.id
                idx = client.id
                weight = self.asymmetric_relation[idx][k]
                # weight = 0.05
                self.personalized_weights.append(weight)

            tot_weights = sum(self.personalized_weights)
            for i, w in enumerate(self.personalized_weights):
                self.personalized_weights[i] = w / tot_weights

            for w, client_model in zip(self.personalized_weights, self.personalized_models):
                for client_k_param, client_param in zip(selected_client.agent_model.parameters(),
                                                        client_model.parameters()):
                    if w == 0:
                        continue
                    client_k_param.data += client_param.data.clone() * w

    def personalized_aggregate(self):
        assert (len(self.uploaded_models) > 0)
        # for k in range(len(self.clients)):
        for k in self.uploaded_ids:
            list_deltaW = torch.zeros(self.personalized_list[0].shape, requires_grad=False).to(self.device)
            for idx in self.uploaded_ids:
                if self.asymmetric_relation[k][idx] == 0:
                    continue
                deltaW = self.uploaded_list[idx] - self.personalized_list[k]
                deltaW = deltaW / (deltaW.norm(p=2) + 1e-12)
                list_deltaW += self.asymmetric_relation[k][idx] * deltaW
            list_oldW = self.params_tolist(self.personalized_models[k])
            list_newW = list_oldW + self.global_lr * list_deltaW
            self.list_load_to_params(self.personalized_models[k], list_newW)

    def receive_models(self):
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(copy.deepcopy(client.agent_model))

    def get_deltaW(self):
        for k in range(len(self.clients)):
            for j in range(len(self.uploaded_models)):
                deltaW = self.params_tolist(self.uploaded_models[j]) - self.params_tolist(self.personalized_models[k])
                deltaW = deltaW / (deltaW.norm(p=2) + 1e-12)
                self.updated_deltaW[k][self.uploaded_ids[j]] = deltaW

    def get_listW(self):
        for k in range(self.num_clients):
            self.personalized_list[k] = self.params_tolist(self.personalized_models[k])
        for k in range(len(self.uploaded_ids)):
            self.uploaded_list[self.uploaded_ids[k]] = self.params_tolist(self.uploaded_models[k])
