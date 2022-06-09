# Reference: https://velog.io/@dldydldy75/

import numpy as np
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision
import time
import copy

EPOCHS = 100
BATCH_SIZE=64
LEARNING_RATE=0.001

DATASET_PATH = "/Users/USER/PycharmProjects/pythonProject2/gp_image"

import torchvision.models as models


def test_eval(model, data_iter, batch_size):
    with torch.no_grad():
        test_loss = 0
        total = 0
        correct = 0
        model.eval()
        for batch_img, batch_lab in data_iter:
            X = batch_img.to(device)
            Y = batch_lab.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == Y).sum().item()
            total += batch_img.size(0)
        val_acc = (100 * correct / total)
        model.train()
    return val_acc



def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(title)
    return ax

if __name__ == '__main__':
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(DATASET_PATH + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(DATASET_PATH + '/test', transform=test_transforms)

    train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    pretrained_model = models.resnet50()


    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pretrained_model = pretrained_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_model.parameters(), lr=LEARNING_RATE)

    total_params = 0
    for param_name, param in pretrained_model.named_parameters():
        if param.requires_grad:
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")
    # Training Phase
    print_every = 1
    print("Start training !")
    # Training loop
    for epoch in range(EPOCHS):
        loss_val_sum = 0
        for batch_img, batch_lab in train_iter:
            X = batch_img.to(device)
            Y = batch_lab.to(device)

            # Inference & Calculate loss
            y_pred = pretrained_model.forward(X)
            loss = criterion(y_pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss

        if ((epoch % print_every) == 0) or (epoch == (EPOCHS - 1)):
            # accr_val = M.test(x_test, y_test, batch_size)
            loss_val_avg = loss_val_sum / len(train_iter)
            accr_val = test_eval(pretrained_model, test_iter, BATCH_SIZE)
            print(f"epoch:[{epoch + 1}/{EPOCHS}] cost:[{loss_val_avg:.3f}] test_accuracy:[{accr_val:.3f}]")

    print("Training Done !")

    test_iter = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    data_iter = iter(test_iter)
    images, labels = next(data_iter)

    inputs, classes = next(iter(train_iter))
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[x for x in classes])

    n_sample = 16
    # sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
    test_x = images[:n_sample]
    test_y = labels[:n_sample]

    with torch.no_grad():
        pretrained_model.eval()
        y_pred = pretrained_model.forward(test_x.type(torch.float).to(device))
        pretrained_model.train()

    y_pred = y_pred.argmax(axis=1)

    plt.figure(figsize=(20, 20))

    test_accuracy = sum([1 for i, j in zip(y_pred, test_y) if i == j]) / n_sample
    print(f"test_accuracy : {test_accuracy:.f}")

    for idx in range(n_sample):
        ax = plt.subplot(4, 4, idx + 1)
        title = f"Predict: {y_pred[idx]}, Label: {test_y[idx]}"
        imshow(test_x[idx], ax, title)

    plt.show()




