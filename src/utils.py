from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report


def kfold_dataset_generator(dataset, batch_size, k=5, shuffle_dataset=True):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size / float(k)))
    if shuffle_dataset:
        np.random.shuffle(indices)
    for fold_no in range(k):
        train_indices = indices[: fold_no * split] + \
            indices[min((fold_no + 1) * split, dataset_size):]
        test_indices = indices[fold_no * split:
                               min((fold_no + 1) * split, dataset_size)]

        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler)

        yield train_loader, test_loader


def get_current_lr(optimizer):
    ''' Get learning rate of optimizer '''
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_classification_report(device, model, test_loader, output_dict=True):
    model.to(device)
    model.eval()

    y_test = []
    y_pred = []
    total = len(test_loader)

    with torch.no_grad():
        for _, (inputs, xc, targets) in enumerate(tqdm(test_loader)):
            inputs, xc, targets = inputs.to(device), xc.to(
                device), targets.to(device)
            outputs = model(inputs, xc)
            outputs = torch.sigmoid(outputs)
            outputs = torch.round(outputs)

            y_test.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    return classification_report(y_test, y_pred, zero_division=0, output_dict=output_dict)
