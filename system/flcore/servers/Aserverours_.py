import copy
import time
from random import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from system.flcore.clients.Aclientours import clientOurs
from system.flcore.servers.serverbase import Server

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from torch.autograd import Function


class OursFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientOurs)
        self.personalized_models = [copy.deepcopy(args.model) for _ in range(self.num_clients)]
        self.uploaded_list = [[] for _ in range(self.num_clients)]
        self.personalized_list = [[] for _ in range(self.num_clients)]
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # Initialize layer-wise asymmetric relation matrix
        num_layers = len(list(args.model.parameters()))
        # print(list(args.model.named_parameters()))
        self.asymmetric_relation = np.array([[[1.0 for _ in range(num_layers)]
                                              for _ in range(self.num_clients)]
                                             for _ in range(self.num_clients)])
        for i in range(self.num_clients):
            for l in range(num_layers):
                self.asymmetric_relation[i][i][l] = 1.0

        self.global_lr = args.global_lr
        self.ar_lr = args.ar_lr
        self.Budget = []
        self.gradients = [[] for _ in range(self.num_clients)]
        self.num_layers = num_layers

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
                # print(client.train_samples,client.id)
                client.train()
                self.gradients[client.id] = client.collect_gradients(self.personalized_models[client.id], False)
                for g in self.gradients[client.id]:
                    g.clamp_(min=-5.0, max=5.0)
            c_t = time.time() - c_t
            print('-' * 25, 'client time', '-' * 25, c_t)
            #############
            self.receive_models()
            self.get_listW()
            #############
            with torch.no_grad():
                if i != 0:
                    trainable_asymmetric_relation = torch.tensor(self.asymmetric_relation, device=self.device)
                    for client in self.selected_clients:
                        for idx in self.uploaded_ids:
                            for l in range(self.num_layers):
                                if trainable_asymmetric_relation[client.id][idx][l] == 0:
                                    continue
                                deltaW = self.uploaded_list[idx][l] - self.personalized_list[client.id][l]
                                deltaW = deltaW / (deltaW.norm(p=2) + 1e-12)
                                # print(self.gradients[client.id][l])
                                gradients_k_idx = self.global_lr * self.gradients[client.id][l].T @ deltaW
                                trainable_asymmetric_relation[client.id][idx][l] = min(max(
                                    trainable_asymmetric_relation[client.id][idx][l] - self.ar_lr * gradients_k_idx,
                                    0.0), 5)
                    self.asymmetric_relation = trainable_asymmetric_relation.cpu().detach().numpy()
            #############
            self.personalized_aggregate()
            #############
            for k in range(self.num_clients):
                self.clients[k].set_parameters(self.personalized_models[k])
            #############
            if i % 10 == 0:
                print("\nGain Relationship Matrix")
                # Print average relation weights across layers for each client pair
                print(pd.DataFrame(self.asymmetric_relation[0][:][:]))
                avg_relations = np.mean(self.asymmetric_relation, axis=2)
                df = pd.DataFrame(avg_relations)
                pd.options.display.float_format = '{:.3f}'.format
                print(df)
            #
            #     # Plot heatmap of average relations
            #     plt.imshow(np.log(avg_relations + 1) / np.log(6), cmap="turbo", vmin=0, vmax=1)  # 设置颜色映射范围
            #     plt.colorbar()
            #     plt.title("Average Relation Matrix Across Layers")
            #     plt.xticks(np.arange(self.num_clients), np.arange(self.num_clients))
            #     plt.yticks(np.arange(self.num_clients), np.arange(self.num_clients))
            #     plt.show()
            #############
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
        return params_list

    def list_load_to_params(self, model, params_list):
        params = list(model.parameters())
        with torch.no_grad():
            for p, new_p in zip(params, params_list):
                p.data.copy_(new_p.view(p.shape))

    def personalized_transit_point(self):
        for selected_client in self.selected_clients:
            # Initialize all parameters to zero
            for param in selected_client.agent_model.parameters():
                param.data.zero_()

            # Calculate personalized weights for each layer
            layer_weights = []  # Will store weights for each layer
            for cid in range(self.num_clients):
                client_weights = []
                k = selected_client.id
                idx = cid  # Assuming client_model has an id attribute

                # Get weights for each layer from asymmetric_relation
                for layer_idx in range(len(self.asymmetric_relation[idx][k])):
                    client_weights.append(self.asymmetric_relation[idx][k][layer_idx])

                layer_weights.append(client_weights)

            # Normalize weights layer-wise
            num_layers = len(layer_weights[0])  # Number of layers in the model
            for layer_idx in range(num_layers):
                # Sum weights for current layer across all clients
                layer_sum = sum(client_weights[layer_idx]
                                for client_weights in layer_weights)

                # Normalize each client's weight for this layer
                for client_idx in range(len(layer_weights)):
                    if layer_sum != 0:
                        layer_weights[client_idx][layer_idx] /= layer_sum

            # Perform layer-wise weighted aggregation
            for client_idx, (w_list, client_model) in enumerate(zip(layer_weights, self.personalized_models)):
                for layer_idx, (client_k_param, client_param) in enumerate(zip(
                        selected_client.agent_model.parameters(),
                        client_model.parameters())):

                    weight = w_list[layer_idx]
                    if weight != 0:
                        client_k_param.data += client_param.data.clone() * weight

    def personalized_aggregate(self):
        assert (len(self.uploaded_models) > 0)

        for k in self.uploaded_ids:
            # Initialize list to store updated parameters for each layer
            updated_params = []

            for l in range(self.num_layers):
                # Initialize delta for current layer
                layer_delta = torch.zeros_like(self.personalized_list[k][l])

                for idx in self.uploaded_ids:
                    if np.mean(self.asymmetric_relation[k][idx]) == 0:  # Skip if average relation is zero
                        continue

                    # Get layer-specific relation weight
                    layer_weight = self.asymmetric_relation[k][idx][l]
                    if layer_weight == 0:
                        continue

                    # Compute delta for current layer
                    deltaW = self.uploaded_list[idx][l] - self.personalized_list[k][l]
                    deltaW = deltaW / (deltaW.norm(p=2) + 1e-12)
                    layer_delta += layer_weight * deltaW

                # Update current layer's parameters
                updated_layer_params = self.personalized_list[k][l] + self.global_lr * layer_delta
                updated_params.append(updated_layer_params)

            # Update the personalized model with all layers' new parameters
            self.list_load_to_params(self.personalized_models[k], updated_params)

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

    def get_listW(self):
        for k in range(self.num_clients):
            self.personalized_list[k] = self.params_tolist(self.personalized_models[k])
        for k in range(len(self.uploaded_ids)):
            self.uploaded_list[self.uploaded_ids[k]] = self.params_tolist(self.uploaded_models[k])
