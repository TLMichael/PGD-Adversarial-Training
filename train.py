import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy_top1

cudnn.benchmark = True


def _model_loop(args, loop_type, loader, model, opt, epoch, writer, adv, plot_gap=np.inf):
    losses = AverageMeter()
    acc_top1 = AverageMeter()

    is_train = (loop_type == 'Train')
    model = model.train() if is_train else model.eval()
    classifier_criterion = nn.CrossEntropyLoss()

    iterator = tqdm(enumerate(loader), total=len(loader))
    for i, (inp, target) in iterator:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits, inp_adv = model(inp, make_adv=adv, target=target)

        classifier_loss = classifier_criterion(logits, target)
        losses.update(classifier_loss.item(), inp.size(0))

        prec = accuracy_top1(logits, target)
        acc_top1.update(prec, inp.size(0))

        if is_train:
            opt.zero_grad()
            classifier_loss.backward()
            opt.step()
        if i % plot_gap == 0 and writer is not None:
            if adv:
                input_output = [inp[:8], inp_adv[:8], inp[:8] - inp_adv[:8],
                                inp[8:16], inp_adv[8:16], inp[8:16] - inp_adv[8:16]]
                input_output = torch.cat(input_output)
                input_output = make_grid(input_output, nrow=8, normalize=True, scale_each=True)
                writer.add_image(loop_type + '_adversarial', input_output, epoch + i // plot_gap)

        desc = ('{} Epoch:{} | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(loop_type, epoch, losses.avg, acc_top1.avg))
        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'accuracy']
        vals = [losses, acc_top1]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg, epoch)

    return losses.avg, acc_top1.avg


def train_model(args, train_loader, val_loader, model, optimizer, schedule, writer):
    start_epoch, best_loss, best_acc = 0, np.inf, 0

    for epoch in range(start_epoch, args.max_epoch):
        train_loss, train_acc = _model_loop(args, 'Train', train_loader, model, optimizer, epoch, writer, args.adv_train)

        snapshot = {
            'model': model.state_dict(),
            'epoch': epoch,
        }

        last_epoch = (epoch == (args.max_epoch - 1))
        should_log = (epoch % args.log_gap == 0)
        if should_log or last_epoch:
            with torch.no_grad():
                test_loss, test_acc = _model_loop(args, 'Val', val_loader, model, None, epoch, writer, adv=False)
            if args.adv_train:
                adv_loss, adv_acc = _model_loop(args, 'Val', val_loader, model, None, epoch, writer, adv=True)

            our_acc = adv_acc if args.adv_train else test_acc
            is_best = our_acc > best_acc
            best_acc = max(our_acc, best_acc)

            torch.save(snapshot, args.model_save_path)
            if is_best:
                torch.save(snapshot, args.model_best_path)
        schedule.step()
    return model


def eval_model(args, test_loader, model, writer):
    test_loss, test_acc = _model_loop(args, 'Val', test_loader, model, None, 0, writer, adv=False, plot_gap=args.plot_gap)
    adv_loss, adv_acc = _model_loop(args, 'Val', test_loader, model, None, 0, writer, adv=True, plot_gap=args.plot_gap)

