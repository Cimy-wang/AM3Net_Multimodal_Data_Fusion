#! -*- coding: utf-8 -*-

# In[1]:  import
from torch.utils.data import DataLoader, TensorDataset
from visdom import Visdom
from util_torch import *
import numpy as np
import scipy.io as scio
from torch import optim
from loss import MarginLoss
vis = Visdom(env='main')
# In[9]: Def super-parameter
iter = 5
BATCH_SIZE = 256
EPOCHS = 100
windowSize = 25
numComponents = 15

# In[10]: prepossessing the data
labels = scio.loadmat('trento_data.mat')['ground']
data_hsi_o = np.array(scio.loadmat('trento_data.mat')['HSI_data'])
data_lidar = np.array(scio.loadmat('trento_data.mat')['LiDAR_data'])
data_lidar = np.reshape(data_lidar, (data_lidar.shape[0], data_lidar.shape[1], 1))
num_classes = np.max(np.max(labels))
result_index = np.zeros((num_classes + 5, iter))

data_hsi, _ = applyPCA(data_hsi_o, numComponents=numComponents)
data_lidar, _ = applyPCA(data_lidar, numComponents=1)

patchesData_hsi, _ = createPatches(data_hsi, labels, windowSize=windowSize)
patchesData_lidar, patchesLabels = createPatches(data_lidar, labels, windowSize=windowSize)

patchesLabels = patchesLabels.astype(np.int32)

train_sample = (129, 125, 105, 154, 184, 122)
validate_sample = (80, 80, 80, 80, 80, 80)

trainIndex, valIndex, testIndex = random_sample(train_sample, validate_sample, patchesLabels)

true_label = patchesLabels[testIndex]

x_train_hsi, x_test_hsi, x_val_hsi = np.array(patchesData_hsi[trainIndex, :, :, :]).astype(np.float32), \
                                     np.array(patchesData_hsi[testIndex, :, :, :]).astype(np.float32), \
                                     np.array(patchesData_hsi[valIndex, :, :, :]).astype(np.float32)
x_train_lidar, x_test_lidar, x_val_lidar = np.array(patchesData_lidar[trainIndex, :, :, :]).astype(np.float32), \
                                           np.array(patchesData_lidar[testIndex, :, :, :]).astype(np.float32), \
                                           np.array(patchesData_lidar[valIndex, :, :, :]).astype(np.float32)

y_train, y_val, y_test = np.array(patchesLabels[trainIndex] - 1).astype(np.float32), \
                         np.array(patchesLabels[valIndex] - 1).astype(np.float32), \
                         np.array(patchesLabels[testIndex] - 1).astype(np.float32)

x_train_hsi_torch, x_val_hsi_torch, x_test_hsi_torch = torch.from_numpy(x_train_hsi), \
                                                       torch.from_numpy(x_val_hsi), \
                                                       torch.from_numpy(x_test_hsi)
x_train_lidar_torch, x_val_lidar_torch, x_test_lidar_torch = torch.from_numpy(x_train_lidar), \
                                                             torch.from_numpy(x_val_lidar), \
                                                             torch.from_numpy(x_test_lidar)

y_train_torch, y_val_torch, y_test_torch = torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test)

train_loader = DataLoader(dataset=TensorDataset(x_train_hsi_torch, x_train_lidar_torch, y_train_torch), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=TensorDataset(x_val_hsi_torch, x_val_lidar_torch, y_val_torch), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=TensorDataset(x_test_hsi_torch, x_test_lidar_torch, y_test_torch), batch_size=BATCH_SIZE, shuffle=True)

# In[11]: train the modal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = MarginLoss(size_average=False, loss_lambda=0.25)
for itera in range(1, iter + 1):
    model = octfusion_multi_adder_1(in_channels_1=data_hsi.shape[2], in_channels_2=data_lidar.shape[2],
                                    out_channels=num_classes).to(DEVICE)

    # model = torch.load('best_model.pt')

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    model, train_time = train(model, criterion, DEVICE, train_loader, optimizer, EPOCHS, vis, val_loader, itera)
    # In[12]: test the modal

    test_acc_temp, test_loss_temp, y_pred, target, test_time = test(model, DEVICE, test_loader)

    classification, confusion, accuracy_matrix = reports(y_pred, target)

    del model

