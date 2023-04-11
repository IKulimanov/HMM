from random import sample

import torch
import warnings
import torch.nn as nn
import numpy as np


from RBM import RBM
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim import Adam, SGD


class DBN(nn.Module):
    def __init__(self, hidden_units, visible_units=512, output_units=1, k=2,
                 learning_rate=1e-5, learning_rate_decay=False,
                 increase_to_cd_k=False, device='cpu'):
        super(DBN, self).__init__()

        self.n_layers = len(hidden_units)
        self.rbm_layers = []
        self.rbm_nodes = []
        self.device = device
        self.is_pretrained = False
        self.is_finetune = False

        # Creating RBM layers
        for i in range(self.n_layers):
            if i == 0:
                input_size = visible_units
            else:
                input_size = hidden_units[i - 1]
            rbm = RBM(visible_units=input_size, hidden_units=hidden_units[i],
                      k=k, learning_rate=learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      increase_to_cd_k=increase_to_cd_k, device=device)

            self.rbm_layers.append(rbm)

        self.W_rec = [self.rbm_layers[i].weight for i in range(self.n_layers)]
        self.bias_rec = [self.rbm_layers[i].h_bias for i in range(self.n_layers)]

        for i in range(self.n_layers):
            self.register_parameter('W_rec%i' % i, self.W_rec[i])
            self.register_parameter('bias_rec%i' % i, self.bias_rec[i])

        self.bpnn = torch.nn.Linear(hidden_units[-1], output_units).to(self.device)

    def forward(self, input_data):

        v = input_data.to(self.device)
        hid_output = v.clone()
        for i in range(len(self.rbm_layers)):
            hid_output, _ = self.rbm_layers[i].to_hidden(hid_output)
        output = self.bpnn(hid_output)
        return output

    def reconstruct(self, input_data):

        h = input_data.to(self.device)
        p_h = 0
        for i in range(len(self.rbm_layers)):
            # h = h.view((h.shape[0], -1))
            p_h, h = self.rbm_layers[i].to_hidden(h)

        for i in range(len(self.rbm_layers) - 1, -1, -1):
            # h = h.view((h.shape[0], -1))
            p_h, h = self.rbm_layers[i].to_visible(h)
        return p_h, h

    def pretrain(
            self, x, epoch=50, batch_size=10):

        hid_output_i = torch.tensor(x, dtype=torch.float, device=self.device)

        for i in range(len(self.rbm_layers)):
            print("Training rbm layer {}.".format(i + 1))

            dataset_i = TensorDataset(hid_output_i)
            dataloader_i = DataLoader(dataset_i, batch_size=batch_size, drop_last=False)

            self.rbm_layers[i].train_rbm(dataloader_i, epoch)
            hid_output_i, _ = self.rbm_layers[i].forward(hid_output_i)

        self.is_pretrained = True
        return

    def pretrain_single(self, x, layer_loc, epoch, batch_size):
        if layer_loc > len(self.rbm_layers) or layer_loc <= 0:
            raise ValueError('Layer index out of range.')
        ith_layer = layer_loc - 1
        hid_output_i = torch.tensor(x, dtype=torch.float, device=self.device)

        for ith in range(ith_layer):
            hid_output_i, _ = self.rbm_layers[ith].forward(hid_output_i)

        dataset_i = TensorDataset(hid_output_i)
        dataloader_i = DataLoader(dataset_i, batch_size=batch_size, drop_last=False)

        self.rbm_layers[ith_layer].train_rbm(dataloader_i, epoch)
        hid_output_i, _ = self.rbm_layers[ith_layer].forward(hid_output_i)
        return

    def finetune(self, x, y, epoch, batch_size, loss_function, optimizer, shuffle=False):

        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        dataset = FineTuningDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)

        print('Begin fine-tuning.')
        for epoch_i in range(1, epoch + 1):
            total_loss = 0
            for batch in dataloader:
                input_data, ground_truth = batch
                input_data = input_data.to(self.device)
                ground_truth = ground_truth.to(self.device)
                output = self.forward(input_data)
                loss = loss_function(ground_truth, output)
                # print(list(self.parameters()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Display train information
            if total_loss >= 1e-4:
                disp = '{2:.4f}'
            else:
                disp = '{2:.3e}'

            print(('Epoch:{0}/{1} -rbm_train_loss: ' + disp).format(epoch_i, epoch, total_loss))

        self.is_finetune = True

        return

    def predict(self, x, batch_size, shuffle=False):
        if not self.is_pretrained:
            warnings.warn("Hasn't pretrained DBN model yet. Recommend "
                          "run self.pretrain() first.", RuntimeWarning)

        if not self.is_pretrained:
            warnings.warn("Hasn't finetuned DBN model yet. Recommend "
                          "run self.finetune() first.", RuntimeWarning)
        y_predict = torch.tensor([])

        x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size, shuffle)
        i = 0
        with torch.no_grad():
            for batch in dataloader:
                y = self.forward(batch[0])
                y_predict = torch.cat((y_predict, y.cpu()), 0)

        return y_predict.flatten()

    def generate_visible_samples(self, k=15, number_samples=100, indices=None, mean_field=True):

        print("Gibbs sampling at inner RBM: %s" % str(len(self.models)))
        samples = self.top_samples
        if indices is None:
            new_data = [self.rbm[len(self.rbm) - 1].gibbs_sampling(img, k) for img in sample(samples, number_samples)]
        else:
            samples = [samples[i] for i in indices]
            new_data = [self.rbm[len(self.rbm) - 1].gibbs_sampling(img, k) for img in samples]
        for i in reversed(range(len(self.rbm) - 1)):
            print("Downward propagation at model: %s" % str(i + 1))
            if mean_field:
                new_data = [self.rbm[i].prop_down(img) for img in new_data]
            else:
                new_data = [self.rbm[i].random_sample(self.rbm[i].prop_down(img)) for img in new_data]
        return new_data


class FineTuningDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
