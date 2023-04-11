import torch
import numpy as np
import matplotlib.pyplot as plt

from DBN import DBN
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error

batch_size = 8
epoch_pretrain = 100
epoch_finetune = 200
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam

with open("dataset/sin(x+1).txt", 'r') as f:
    dataset = np.genfromtxt(f)

scaler = StandardScaler()
dataset_norm = scaler.fit_transform(dataset.reshape(-1, 1)).flatten()
dataset_list = []
for i in range(len(dataset) - input_length - output_length):
    dataset_list.append(dataset_norm[i:i + input_length + output_length])
dataset_list = np.array(dataset_list)
trainset = dataset_list[:int(len(dataset_list) * (1 - test_percentage))]
testset = dataset_list[int(len(dataset_list) * (1 - test_percentage)):]

x_train = trainset[:, :-1]
y_train = trainset[:, -1:]
x_test = testset[:, :-1]
y_test = testset[:, -1:]
#z_test = np.zeros([249, 1])


print('x_train.shape:' + str(x_train.shape))
print('y_train.shape:' + str(y_train.shape))
print('x_test.shape:' + str(x_test.shape))
print('y_test.shape' + str(y_test.shape))

# Build model
dbn = DBN(hidden_units, input_length, output_length, device=device)

# Train model
dbn.pretrain(x_train, epoch=epoch_pretrain, batch_size=batch_size)
dbn.finetune(x_train, y_train, epoch_finetune, batch_size, loss_function,
             optimizer(dbn.parameters()))

# Make prediction and plot

y_predict = dbn.predict(x_test, batch_size)

y_real = scaler.inverse_transform(x_test.reshape(-1, 1)).flatten()
y_predict = scaler.inverse_transform(y_predict.reshape(-1, 1)).flatten()
plt.figure(1)
#plt.plot(y_real, label='real')
plt.plot(y_predict, label='prediction')
plt.xlabel('MSE Error: {}'.format(mean_squared_error(y_real, y_predict)))
plt.legend()
plt.title('Prediction result')
plt.show()
