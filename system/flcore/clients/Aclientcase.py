import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from system.flcore.clients.clientbase import Client


class clientOurs(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.trainloader = self.load_train_data()
        self.testloader = self.load_test_data()
        self.agent_model = copy.deepcopy(args.model)
        self.optimizer = torch.optim.SGD(self.agent_model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        # self.model.to(self.device)
        # self.trainloader = self.load_train_data()
        self.agent_model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.agent_model(x)
                loss = self.loss(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.agent_model.parameters(), max_norm=10)
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_gradients(self, temp_model, cat=True):
        loss = 0
        cnt = 0
        for i, (x, y) in enumerate(self.load_train_data(32)):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            output = temp_model(x)
            loss += self.loss(output, y)
            cnt += 1
            break
        loss = loss / cnt
        loss.backward()
        gradients = []
        for param in temp_model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().clone().view(-1))
            else:
                gradients.append(None)
        if cat:
            gradients = torch.cat(gradients)
        return gradients
