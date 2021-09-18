import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models


def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)  # (len(labels), K)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class: (len(labels), K-1)
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels


def class_prior(complementary_labels):
    # p(\bar{y})
    return np.bincount(complementary_labels) / len(complementary_labels)


def prepare_data(dataset, batch_size):
    if dataset == "mnist":
        ordinary_train_dataset = dsets.MNIST(root='./data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor())
        input_dim = 28 * 28
    elif dataset == "cifar10":
        ordinary_train_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.CIFAR10(root='./data/CIFAR10', train=False, transform=transforms.ToTensor())
        input_dim = 3 * 32 * 32
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=True)
    num_classes = len(ordinary_train_dataset.classes)
    return full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes, input_dim


def prepare_train_loaders(full_train_loader, batch_size, ordinary_train_dataset):
    for i, (data, labels) in enumerate(full_train_loader):
        K = torch.max(labels)+1  # K is number of classes, full_train_loader is full batch
        bs = labels.size(0)
    complementary_labels = generate_compl_labels(labels)
    x_to_tls = {i: -1 for i in range(bs)}
    x_to_mcls = {i: set() for i in range(bs)}
    for idx, tl in enumerate(labels.tolist()):
        x_to_tls[idx] = tl
    for idx, cl in enumerate(complementary_labels.tolist()):
        x_to_mcls[idx].add(cl)
    ccp = class_prior(complementary_labels)
    id = torch.arange(bs)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).long(), labels, id)
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)
    return ordinary_train_loader, complementary_train_loader, ccp, x_to_mcls, x_to_tls
