import os
import sys
#print(sys.path)
#sys.path.append('.')
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set#, get_test_set
from utils import Logger
from train import train_epoch
from train_multiple_losses import train_epoch as train_mult_loss_epoch

from validation import val_epoch
from validation_multiple_losses import val_epoch as val_mult_loss_epoch

from video_level_accuracy import val_epoch_video_level
import evaluate 
def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []

    parameters = []
    for k, v in model.named_parameters():
        if "classifier" in k :
            parameters.append({'params': v, 'lr': 0.01})
            break
        elif "conv5" in k:
            parameters.append({'params': v, 'lr': 0.001})
        else:
            parameters.append({'params': v, 'lr': 0.01})

    return parameters

if __name__ == '__main__':
    torch.backends.cudnn.benchmark=True

    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    input_config = {}
    input_config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    input_config['std'] = [1 / (.0167 * 255)] * 3
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    
    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    #print(model)
    #print(model.parameters())
    
    criterion = nn.CrossEntropyLoss()
    
    if not opt.no_cuda:
        model.cuda()
        model = nn.DataParallel(model)
    if not opt.no_cuda:
        criterion = criterion.cuda()
    if not opt.mult_loss: #and not opt.mv and not opt.opticflow:
        print("loading weights")     
        pretrain = torch.load(opt.pretrain_path) 
        
        if opt.n_finetune_classes != 0:
            #parameters = get_fine_tuning_parameters(model, 4)
            print("cleaning the classifier")
            #print(pretrain['state_dict'])
            if 'resnet' in opt.model or 'resnext' in opt.model:
                del pretrain['state_dict']['module.fc.weight'] 
                del pretrain['state_dict']['module.fc.bias'] 
            else:
                del pretrain['state_dict']['module.classifier.weight'] 
                del pretrain['state_dict']['module.classifier.bias'] 
        model.load_state_dict(pretrain['state_dict'],strict= True)#strict=True)
        
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])

        normalize = Normalize(mean=input_config['mean'], std=input_config['std'])
        if not opt.mfnet_st:
            spatial_transform = Compose([
                crop_method,
                RandomHorizontalFlip(),
                ToTensor(opt.norm_value), norm_method
            ])
        else:
            spatial_transform = Compose([
                                             Scale((256,256)),
                                             crop_method, # insert a resize if needed
                                             RandomHorizontalFlip(),
                                             #transforms.RandomHLS(vars=[15, 35, 25]),
                                             ToTensor(),
                                             normalize,
                                          ])

        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
        
    if not opt.no_val:
        if not opt.mfnet_st:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value), norm_method
            ])
        else:
            normalize = Normalize(mean=input_config['mean'], std=input_config['std'])
            spatial_transform = Compose([
                                             Scale((256,256)),
                                             MultiScaleRandomCrop(opt.scales, opt.sample_size),
                                             ToTensor(),
                                             normalize,
                                            ])

        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        #print(len(val_loader))
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])
    
    
    
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            if opt.mult_loss:
                train_mult_loss_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
            else:
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            if opt.video_level_accuracy:
                val_acc = val_epoch_video_level(i, val_loader, model, opt,criterion)
            elif opt.mult_loss:
                validation_loss = val_mult_loss_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            else:
                validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
    if opt.test:
        if not opt.mfnet_st:
            spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value), norm_method
            ])
        else:
            normalize = Normalize(mean=input_config['mean'], std=input_config['std'])
            spatial_transform = Compose([
                                             Scale((256,256)),
                                             MultiScaleRandomCrop(opt.scales, opt.sample_size),
                                             ToTensor(),
                                             normalize,
                                          ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        evaluate.test_net(model, test_loader, opt)
    