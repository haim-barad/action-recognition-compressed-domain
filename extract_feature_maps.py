import os
import sys
import json
import numpy as np
import argparse
import torch
import subprocess
from torch import nn
from  mfnet_3d_multiple_exits import MFNET_3D
#from  mfnet_3d import MFNET_3D
from opts import parse_opts
from model import generate_model
from torchvision import transforms
from mean import get_mean, get_std
import random
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
import cv2
from dataset import get_training_set, get_validation_set

def writingVideo(images,channel_path,frame_size,channel):
    # Create a VideoCapture object
    #path = os.path.join("/kinetics2/kinetics2/featuremaps-ucf101/UCF-101/",channel_path)
    if not os.path.exists(channel_path):    
        video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
        #video_FourCC = cv2.VideoWriter_fourcc(*"MJPEG")
        writer = cv2.VideoWriter(channel_path,video_FourCC, 10, (frame_size, frame_size), 0)
        for image in images:
            writer.write(np.array(image))
        writer.release()
        #stop
        
def tensor2img(tensors,dir_path):
    tensor_2_image = transforms.ToPILImage()  # Converts to Image
    channels = tensors.shape[0]
   # listofchannels
    for channel in range(channels):
        tensor = tensors[channel:channel+1,:,:,:]
        tensor = tensor.permute(1, 0, 2, 3)  # Change axises -> 16,16,56,56
        frame_size = tensor.shape[-1]
        tensor = tensor.detach().cpu()#.numpy()
        
        images = [tensor_2_image(im) for im in tensor]  # Convert images
      

        channel_path = os.path.join(dir_path, 'channel_' + str(channel) + '.mp4')        
       
        writingVideo(images,channel_path,frame_size,channel)
       
    return tensor
    
    
def features_in_video_out(outputs, index, CompresOrStandard,output_path):  ##Example : subpath = 'SalsaSpin/v_SalsaSpin_g05_c01.mp4'
    vid_dir = "vid_" + str(index) + CompresOrStandard
    dir_path = os.path.join(output_path,vid_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for exit_num in range(len(outputs)):#looping on number of exits
        print("exit:",exit_num)
        exit_dir_path = os.path.join(dir_path ,'exit-' + str(exit_num))
        #save_path = featuremap_level_dir_path + 'exit' + str(exit_num) + '.json'
        if not os.path.exists(exit_dir_path):
            print(exit_dir_path,"not exists, creating...")
            os.mkdir( exit_dir_path )
            #else:
            #    if len(os.listdir(featuremap_level_dir_path) ) != 0: ##if he exists check if the folder is empty, if does, continue normaly
            #        break  
            tensor = outputs[exit_num]
            featuremap = tensor2img(tensor,exit_dir_path)
            #subprocess.call(['rm', '-rf',featuremap_level_dir_path])
            
            #torch.save(featuremap,save_path)
            
def write_input_video(input,input_dir_path,CompresOrStandard):
    print("writing the input video to :",input_dir_path)
    if not os.path.exists(input_dir_path):
        os.mkdir(input_dir_path)
    tensor_2_image = transforms.ToPILImage()
    #tensor = input
    #print(tensor.shape)
    tensor = torch.squeeze(input)
    tensor = input.permute(1, 0, 2, 3)
    frame_size = tensor.shape[-1]
   #print("before traslate tensor2image, shape is:",tensor.shape)
    tensor = tensor.detach().cpu()
    images = [tensor_2_image(im) for im in tensor]
    for j,image in enumerate(images):
        #cv2.cvtColor(numpy.array(frame), cv2.COLOR_RGB2BGR)
        end = "image" + str(j) + '.jpg'
        path = os.path.join(input_dir_path,end)

        
        path = os.path.join(input_dir_path,end)
        image.save(path,"JPEG",icc_profile=image.info.get('icc_profile'))
    #video_end = "input_video.mp4"
    #video_path =  path = os.path.join(input_dir_path,video_end)

    #if CompresOrStandard == 'compressed':
#        subprocess.call(['ffmpeg','y','vcodec','rawvideo','s','qvga','start_number','0','f','','','','','','',
    #    subprocess.call(['ffmpeg','framerate', '20','pattern_type','codec:v','mpeg4', 'glob', '-i', 'featuremaps_output/vid_1compressed/input/image1.jpg','/workspace/output.mp4'])#'/workspace/output.mp4'])
    #else:
    #    subprocess.call(['ffmpeg','framerate', '20','pattern_type','glob', '-i', '/workspace/3D-ResNets-PyTorch/featuremaps_output/vid_1standard/input/*.jpg','/workspace/output.mp4'])#'/workspace/output.mp4'])

        
   #print(images[0].mode)
    
    #writingVideo(images,input_dir_path,frame_size,0) #lsat field is not important
    
if __name__ == '__main__':
    opt = parse_opts()
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.set_defaults(compressed=False)
    parser.add_argument('--',default='',type=str)
    parser.add_argument('--annotation_path_standard',default='',type=str,help='')
    parser.add_argument('--annotation_path_compress',default='',type=str)
    
    parser.add_argument('--dataset',default='',type=str)
    
    ################################################################
    parser.add_argument('--opt.model_depth',default=101,type=int)
    parser.add_argument('--initial_scale',default='',type=str)
    parser.add_argument('--norm_value',default='',type=str)
    ################################################################
    need to delete it from the main opt file::::parser.add_argument(
        '--video_index',
        default=-1,
        type=int,
        help='this will make the loader to load one video which is the video_index that been choosen -- espicielly for featuremap extraction')
    """
    #if opt.root_path != '':
    opt.scales = [opt.initial_scale]
    input_config = {}
    input_config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    input_config['std'] = [1 / (.0167 * 255)] * 3
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)


    torch.manual_seed(opt.manual_seed)

    model_compress = MFNET_3D(opt.n_classes)
    model_standrad = MFNET_3D(opt.n_classes)
    standard_pretrain_path = "/workspace/BackupHalfComprss/ucf-orig-results/save_30.pth"
    if not opt.no_cuda:

        model_compress.cuda()
        model_compress = nn.DataParallel(model_compress)
        pretrain = torch.load(opt.pretrain_path)        
        model_compress.load_state_dict(pretrain['state_dict'],strict=True)
        
        model_standrad.cuda()
        model_standrad = nn.DataParallel(model_standrad)
        pretrain = torch.load(standard_pretrain_path)        
        model_standrad.load_state_dict(pretrain['state_dict'],strict=True)
        
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    

    if not opt.mfnet_st:
        spatial_transform = Compose([
                Scale(opt.sample_size),
                CenterCrop(opt.sample_size),
                ToTensor(opt.norm_value)#, norm_method
            ])
    else:
        normalize = Normalize(mean=input_config['mean'], std=input_config['std'])
        spatial_transform = Compose([
                                             Scale((256,256)),
                                             CenterCrop(opt.sample_size),
                                             ToTensor(),
                                        #     normalize,
                                          ])
                                          
                                          
    temporal_transform = LoopPadding(opt.sample_duration)
    opt.compressed = 1 
    opt.residual_only = 1
    residual_data = get_validation_set(
        opt, spatial_transform, temporal_transform)
    opt.residual_only = 0
    compressed_data = get_validation_set(
        opt, spatial_transform, temporal_transform)
    opt.compressed = 0 ###super important this will take data from normal rgb frames
    standard_data = get_validation_set(
        opt, spatial_transform, temporal_transform)
    
    residual_data_loader = torch.utils.data.DataLoader(
        compressed_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    compressed_data_loader = torch.utils.data.DataLoader(
        compressed_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
        
    standard_data_loader = torch.utils.data.DataLoader(
        standard_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    

    inputs_residual, _ = next(iter(residual_data_loader))
    inputs_compressed, _ = next(iter(compressed_data_loader))
    inputs_standard, _ = next(iter(standard_data_loader))
    print("Got the videos,Let's Rock'nRoll")
    inputs_residual = inputs_residual.to('cuda')
    inputs_compressed = inputs_compressed.to('cuda')
    inputs_standard = inputs_standard.to('cuda')
    

    outputs_compress = model_compress(inputs_compressed)
    outputs_standard = model_standrad(inputs_standard)

    inputs_compressed = inputs_compressed.view(opt.batch_size,3,opt.sample_duration,opt.sample_size,opt.sample_size)
    inputs_standard = inputs_standard.view(opt.batch_size,3,opt.sample_duration,opt.sample_size,opt.sample_size)
    
    one_from_batch = []
    for i in range(1):# we have 6 exits but the last one is not in our interest for this goal
        one_from_batch.append(outputs_compress[i][0,:,:,:,:])
    if opt.residual_only:
        CompresOrStandard = 'residual_only'
    else:
        CompresOrStandard = 'compressed'
    print("running for compressed")
    features_in_video_out(one_from_batch,1,CompresOrStandard, opt.result_path)
    vid_dir = "vid_" + str(1) + CompresOrStandard
    dir_path = os.path.join(opt.result_path,vid_dir)
    input_path = os.path.join(dir_path,'input')
    write_input_video(inputs_compressed[0,:,:,:,:],input_path,CompresOrStandard)
    

    
    one_from_batch = []
    #one_from_batch.append(outputs_standard[0])
    for i in range(4):# we have 6 exits but the last one is not in our interest for this goal
        one_from_batch.append(outputs_standard[i][0,:,:,:,:])
        
    CompresOrStandard = 'standard'
    print("running for standard")
    features_in_video_out(one_from_batch,1,CompresOrStandard, opt.result_path)
    vid_dir = "vid_" + str(1) + CompresOrStandard
    dir_path = os.path.join(opt.result_path,vid_dir)
    input_path = os.path.join(dir_path,'input')
    write_input_video(inputs_standard[0,:,:,:,:],input_path,CompresOrStandard)    