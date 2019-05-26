import torch
from torch.autograd import Variable
import time
import sys
from torch import nn
from utils import AverageMeter, calc_top1, calc_top5, calculate_accuracy,calculate_accuracy_video_level, probabily_accumolator
import math
import os
import json 

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[index] = class_label
        index += 1
    return class_labels_map


def val_epoch_video_level(epoch, data_loader, model,opt,criterion,target_transform=None):

    model.eval()
    batch_size = opt.batch_size
    batch_time = AverageMeter()
    data_time = AverageMeter()

    softmax = nn.Softmax(dim=1)
    end_time = time.time()
    #####################
    results_dict = {}
    probability_dict = {}
    #####################
    index = 0
    print("len:::",len(data_loader))
    for batch, (inputs, targets,video_name) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        
        if not opt.no_cuda:
            targets = targets.cuda()
        targets = Variable(targets)
        
        inputs.cuda()
        with torch.no_grad():
            inputs  = Variable(inputs)
        outputs = model(inputs)
        
        outputs = softmax(outputs)
        


        
        probabily_accumolator(outputs, targets, video_name,probability_dict,opt)
#        calculate_accuracy_video_level(outputs, targets, video_name,results_dict)
        
        
       
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        #print('Epoch: [{0}][{1}/{2}]'.format(
        #          epoch,
        #          batch + 1,
        #          len(data_loader)))
    
    final_result_list=[]
    #index = 0
    class_pred = [0] * opt.n_classes
    class_correct = [0] * opt.n_classes
    
    for (video,values) in probability_dict.items():
    
        listOfProbabilities =  torch.FloatTensor(values['probabilities'])
        target_label = values['target_label']
        class_pred[target_label] += 1
        
        #class_scores[]
        _, pred = listOfProbabilities.topk(1, 0, True)
        
        if pred.item() == target_label.item():
            class_correct[target_label] += 1
            final_result_list.append(1)
        else:   
            final_result_list.append(0)
    video_acc = sum(final_result_list) / len(final_result_list)
    print("video_acc is:",video_acc)
    annotation_dir = opt.annotation_path.rsplit('/', 1)[0]
    classes_list =os.path.join(annotation_dir,"classes.json")
    if opt.dataset == "kinetics":
        classes_list = "kinetics_classes.json"
    classes = load_annotation_data(classes_list)
    dict_labels = get_class_labels(classes)
    
    
    with open("class_scors1.txt",'w') as f:
        for i in  range(opt.n_classes):
            f.write("class number: ")
            f.write(str(dict_labels[i]))
            f.write(" has accuracy of:")
            f.write(str(round(class_correct[i]/class_pred[i],2)))

            
            f.write("\n")
    
    
    #return video_acc
    """   
    avg_list = []
    most_wins = [] 
    for (video, values) in results_dict.items():
        # print(values)
        video_acc = sum(values) / len(values)
        #print("video:",video,", got :",video_acc*100,"%") 
        avg_list.append(video_acc * 100)
     #   if video_acc > 0.5:
      #      most_wins.append(1)
      #  else:
      #      most_wins.append(0)
    video_acc = sum(avg_list) / len(avg_list)
    print("TOP1 (avareging clip level top1) :", video_acc)
    #video_acc = sum(most_wins) / len(most_wins)
    #print("TOP1 (avareging most_wins level top1) :", video_acc)
    """
   
###notice the changes
   # return losses.avg, binary_taken, corrects