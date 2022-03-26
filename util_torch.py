from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from model import *
import random
import copy
import datetime

# In[2]  create the image patches
def createPatches(X, y, windowSize, removeZeroLabels=False):
    # X = X.reshape(X.shape[2], X.shape[0], X.shape[1])
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), 'symmetric')
    zeroPaddedX = zeroPaddedX.reshape(zeroPaddedX.shape[2], zeroPaddedX.shape[0], zeroPaddedX.shape[1])
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], X.shape[2], windowSize, windowSize), dtype='float16')
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype='float16')
    patchIndex = 0
    for c in range(margin, zeroPaddedX.shape[2] - margin):
        for r in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[:, r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def random_sample(train_sample, validate_sample, patchesLabels):
    num_classes = int(np.max(patchesLabels))
    dataList = patchesLabels
    TrainIndex = []
    TestIndex = []
    ValidateIndex = []

    for i in range(num_classes):
        train_sample_temp = train_sample[i]
        validate_sample_temp = validate_sample[i]
        index = np.where(patchesLabels == (i + 1))[0]
        Train_Validate_Index = random.sample(range(0, int(index.size)), train_sample_temp + validate_sample_temp)
        TrainIndex = np.hstack((TrainIndex, index[Train_Validate_Index[0:train_sample_temp]])).astype(np.int32)
        ValidateIndex = np.hstack((ValidateIndex, index[Train_Validate_Index[train_sample_temp:100000]])).astype(np.int32)
        Test_Index = [index[i] for i in range(0, len(index), 1) if i not in Train_Validate_Index]
        TestIndex = np.hstack((TestIndex, Test_Index)).astype(np.int32)

    return TrainIndex, ValidateIndex, TestIndex


# In[3]  apply PCA preprocessing for data sets
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

# In[4]: calculate the classification result
def reports(y_pred, target_1):
    classification = classification_report(target_1, y_pred)
    confusion = confusion_matrix(target_1, y_pred)
    oa = np.trace(confusion) / sum(sum(confusion))
    ca = np.diag(confusion) / confusion.sum(axis=1)
    Pe = (confusion.sum(axis=0) @ confusion.sum(axis=1)) / np.square(sum(sum(confusion)))
    K = (oa - Pe) / (1 - Pe)
    aa = sum(ca) / len(ca)
    List = []
    List.append(np.array(oa)), List.append(np.array(aa)), List.append(np.array(K))
    List = np.array(List)
    accuracy_matrix = np.concatenate((ca, List), axis=0)
    return classification, confusion, accuracy_matrix


# In[5]: Def val
def val(model, val_loader, criterion):
    global acc, acc_best
    model.eval()
    total_correct = 0
    eye = torch.eye(int(max(val_loader.dataset.tensors[2]) + 1)).cuda()
    avg_loss = 0.0
    with torch.no_grad():
        for i, (data_hsi, data_lidar, labels) in enumerate(val_loader):
            data_hsi, data_lidar, labels = Variable(data_hsi).cuda(), Variable(data_lidar).cuda(), Variable(labels).cuda()
            output = model(data_hsi, data_lidar)
            labels = labels.to(torch.int64)
            target_hot = eye[labels]
            avg_loss = criterion(output, target_hot)
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
            acc = float(total_correct) / len(val_loader.dataset.tensors[0])

    avg_loss /= len(val_loader.dataset.tensors[0])
    acc = float(total_correct) / len(val_loader.dataset.tensors[0])

    return acc, avg_loss

# In[5]: Def training
def train(model, criterion, device, train_loader, optimizer, EPOCHS, vis, val_loader, itera=1):
    global best_model
    acc_temp = 0
    epoch_temp = 1
    eye = torch.eye(int(max(train_loader.dataset.tensors[2]) + 1)).cuda()
    start_time_train = datetime.datetime.now()
    for epoch in range(1, EPOCHS + 1):
        start_time = datetime.datetime.now()
        model.train()
        number = 0
        for batch_idx, (data_hsi, data_lidar, target) in enumerate(train_loader):
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data_hsi, data_lidar)
            target = target.to(torch.int64)
            target_hot = eye[target]
            loss = criterion(output, target_hot)
            loss.backward()
            optimizer.step()
            output = output.argmax(dim=1)
            number += output.eq(target).float().sum().item()
            cur_time = datetime.datetime.now()

        val_acc, avg_loss = val(model, val_loader, criterion)
        if acc_temp <= val_acc:
            print('Best_Val_Value changed: from %f to %f;' % (acc_temp, val_acc), end="\t")
            epoch_temp = epoch
            acc_temp = val_acc
            best_model = copy.deepcopy(model)
            print('Best Classification Accuracy %f， Best Classification loss %f； Best Epoch： %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")
        else:
            print('Best Classification Accuracy %f， Best Classification loss %f； Best Epoch： %d' % (
            acc_temp, avg_loss, epoch_temp), end="\n")
        vis.line(Y=[[number / len(train_loader.dataset), val_acc]],
                 X=[epoch],
                 win='acc {}'.format(itera),
                 opts=dict(title='acc', legend=['acc', 'val_acc']),
                 update='append')

        model = best_model
    end_time_train = datetime.datetime.now()
    print('||======= Train Time for % s' % (end_time_train - start_time_train), '======||')
    return model, (end_time_train - start_time_train).total_seconds()

# In[6]: Def test
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    target_1 = []
    # torch.cuda.synchronize()
    start_time_test = datetime.datetime.now()
    with torch.no_grad():
        for (data_hsi, data_lidar, target) in test_loader:
            data_hsi, data_lidar, target = data_hsi.to(device), data_lidar.to(device), target.to(device)
            target = target.to(torch.int64)
            output = model(data_hsi, data_lidar)
            y_pred_temp = output.max(1, keepdim=True)[1]
            correct += y_pred_temp.eq(target.view_as(y_pred_temp)).sum().item()
            y_pred_temp_1 = y_pred_temp.data.cpu().numpy()
            target_temp_1 = target.data.cpu().numpy()
            y_pred.extend(y_pred_temp_1)
            target_1.extend(target_temp_1)

        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(1, y_pred.size)
        y_pred = np.array(y_pred).astype(np.float32)
        y_pred = y_pred[0]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc_temp = format(100. * correct / len(test_loader.dataset))
    # test_acc.append(test_acc_temp)
    test_loss_temp = format(test_loss)
    end_time_test = datetime.datetime.now()
    print('||======= Test Time for % s' % (end_time_test - start_time_test), '======||')
    return test_acc_temp, test_loss_temp, y_pred, target_1, (end_time_test - start_time_test).total_seconds()


