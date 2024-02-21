import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import transforms as own_transforms
from resortit import resortit
from config import cfg
from torch.autograd import Variable
import argparse
import os
from metric import SegmentationMetric
from BiSeNet.build_BiSeNet import BiSeNet
import torch
import tqdm
import numpy as np
from loading_data import loading_data
from PIL import Image
import pandas as pd
import optuna


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 0.002, 0.025)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-4)
    momentum = 0.9  # or 0.8

    # Model
    model = BiSeNet(5, 'resnet18').to(device)

    # Trainer execution
    val_loss, val_accuracy = trainer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        epochs=15,
        net=model
    )

    # Optimization based on the minimization of validation loss and validation accuracy
    return val_loss - val_accuracy


def train(model, dataloader_train, optimizer, loss_func):
    model.train()
    loss_record = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (data, label) in enumerate(dataloader_train):
        if torch.cuda.is_available():
            data = data.to(device)
            label = label.to(device)
        output, output_sup1, output_sup2 = model(data)
        output = torch.squeeze(output)
        output_sup1 = torch.squeeze(output_sup1)
        output_sup2 = torch.squeeze(output_sup2)
        loss1 = loss_func(output, label)
        loss2 = loss_func(output_sup1, label)
        loss3 = loss_func(output_sup2, label)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    loss_train_mean = np.mean(loss_record)
    return loss_train_mean


def validate(val_loader, net, loss_f):
    net.eval()
    iou_ = 0.0
    num_classes = 5
    iou_sum_classes = [0.0] * num_classes
    label_trues, label_preds = [], []
    val_loss = []

    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        val_loss.append(loss_f(outputs, labels))
        label_trues.append(labels.cpu().numpy())
        label_preds.append(outputs.argmax(dim=1).cpu().numpy())
        # if vi < 10:  # only do this for the first 10 examples
        # pred_mask = outputs.argmax(dim=1).cpu().numpy()
        # colored_mask = colorize_mask(pred_mask[0])  # colorize the first mask in the batch
        # colored_mask.show()  # display the mask
    acc_cls, mean_iu = scores(label_trues, label_preds, num_classes)
    val_loss = sum(val_loss) / len(val_loss)
    return acc_cls, mean_iu, val_loss


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print(f"Mean Acc: {acc_cls}   Mean IoU: {mean_iu}   Classwise Mean IoU: {iu}")
    return acc_cls, mean_iu


def trainer(
        # lets define the basic hyperparameters
        learning_rate,
        weight_decay,
        momentum,
        epochs,
        net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # now we load the data in three splits train, test and validation
    train_loader, val_loader, test_loader = loading_data()
    # Moving the resnet to gpu device if it is available
    # defining the optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # defining the loss function
    weights = [0.5, 0.7, 0.7, 0.9, 0.7]
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
    # finaly training the model

    # In order to save the accuracy and loss we use a list to save them in each epoch
    val_loss_list = []
    val_miou = []
    val_accuracy_list = []

    for e in range(epochs):
        print('training epoch number {:.2f} of total epochs of {:.2f}'.format(e, epochs))
        train_loss = train(net, train_loader, optimizer, loss_function)
        print('Epoch: {:d}'.format(e + 1))
        print('\t Training loss {:.5f}'.format(train_loss))
        acc_cls, mean_iu, loss = validate(val_loader, net, loss_function)
        val_accuracy_list.append(acc_cls)
        val_miou.append(mean_iu)
    df = pd.DataFrame()
    df['mean_acc'] = val_accuracy_list
    df['mean_iou'] = val_miou
    df.to_csv('/content/training_performance.csv', index=False)
    print("Testing on the test set:")
    val_accuracy, mean_iu, val_loss = validate(test_loader, net)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definizione degli iperparametri METTERE I VALORI DI LR E WD
learning_rate = 0.025  # 0.1
weight_decay = 1e-4
momentum = 0.9
# Creazione del modello
model = BiSeNet(5, 'resnet18').to(device)

return val_loss, val_accuracy

study = optuna.create_study()
# Iteration with different trials over the sampled parameters
study.optimize(objective, n_trials=5)
# Best parameters between trials
print(study.best_params)
