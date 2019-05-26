import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    #print(len(data_loader))

    end_time = time.time()
    print("len:",len(data_loader))
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
    
        if not opt.no_cuda:
            targets = targets.cuda()
            if opt.mv:
                clip    = inputs[0].cuda()
                mvclip  = inputs[1].cuda()
            else:
                inputs  = inputs.cuda()
        with torch.no_grad():
            targets = Variable(targets)
        #    if opt.mv:
        #        clip = Variable(clip)
        #        mvclip = Variable(mvclip)
        #        outputs = model(clip,mvclip)
         #   else:
            inputs = Variable(inputs)
            outputs = model(inputs)
    
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        if opt.mv:
            losses.update(loss.item(), inputs[0].size(0))
            accuracies.update(acc, inputs[0].size(0))  
        else:
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return losses.avg
