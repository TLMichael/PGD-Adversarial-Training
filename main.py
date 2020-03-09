import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import argparse
import os
import os.path as osp
from pprint import pprint

from train import train_model
from model import WideResNet
from attacker import AttackerModel


def main():
    parser = argparse.ArgumentParser(description='PGD based adversarial training')
    args = parser.parse_args()

    # Model options
    args.adv_train = True

    # Training options
    args.dataset = 'cifar10'
    args.batch_size = 128
    args.max_epoch = 200
    args.lr = 0.1
    args.lr_step = 0.1
    args.lr_milestones = [100, 150]
    args.log_gap = 5

    # Attack options
    args.random_start = True
    args.step_size = 2.0 / 255
    args.epsilon = 8.0 / 255
    args.num_steps = 7
    args.targeted = False

    # Miscellaneous
    args.data_path = '~/datasets/CIFAR10'
    args.result_path = './results/classifier'
    args.tensorboard_path = './results/classifier/tensorboard/train'
    args.model_save_path = osp.join(args.result_path, 'model.latest')
    args.model_best_path = osp.join(args.result_path, 'model.best')

    if not osp.exists(args.result_path):
        os.makedirs(args.result_path)
    
    pprint(vars(args))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    classifier = WideResNet(depth=28, num_classes=10, widen_factor=2)
    model = AttackerModel(classifier, vars(args))
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
    schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)

    writer = SummaryWriter(args.tensorboard_path)
    # writer = None

    train_model(args, train_loader, val_loader, model, optimizer, schedule, writer)


if __name__ == '__main__':

    main()
