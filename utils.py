import csv
import os
import torch
from operator import add
import numpy.random as random
import torch.nn.functional as F
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as input_file:
            value = float(input_file.read().rstrip('\n\r'))
    
    else:
        value = -1
    return value

def calc_total_top1(outputs, targets,opt,binary_taken,correct):

    batch_size = targets.size(0)


    total4batch = 0
    for batch in range(batch_size):
        exit_taken = False
        for exit in range(opt.num_exits-1):
            exit_taken = binary_taken[batch*(opt.num_exits-1)+exit] #is this exit taken of not?!
            was_it_correct = correct[batch][exit]
            if exit_taken:
                opt.exit_taken[exit] += 1
                if  was_it_correct:
                    total4batch += 1
                    opt.true_positive[exit] +=1
                else:
                    opt.false_positive[exit] += 1 #I was tricked of thinking that I have the correct result
                break
            else:
                if was_it_correct:
                    opt.true_negative[exit] += 1 # I had the correct result , But I didn't trust my exit!
        exit = opt.num_exits - 1
        was_it_correct = correct[batch][exit]        
        if not exit_taken:
            if was_it_correct:
                opt.exit_taken[exit] += 1
                total4batch += 1
                
    accuracy = total4batch/batch_size
    return accuracy         
        
    #Total accuracy: amount of time I succsedded in the early exit /the total amount of time I took this exit -> tp /tp+fp
    
    #print ("correct shape",correct,correct.shape)
    #n_correct_elems = correct.float().sum().item()
    
   # return n_correct_elems / batch_size

    
    
def calc_top1(outputs, targets,opt):

    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    
    correct = pred.eq(targets.view(1, -1))
    #print ("correct shape",correct,correct.shape)
    
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size, correct


def calc_top5(outputs, targets,opt):
    batch_size = targets.size(0)

    _, pred = outputs.topk(5, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def probabily_accumolator(outputs, targets,video_name,probabily_dict,opt):
    batch_size = outputs.size(0)
   
    for i in range(batch_size):
        #_, pred = outputs[i].topk(1, 0, True)
        #print(outputs[i])
        if video_name[i] not in probabily_dict:
            init_list = [0] * opt.n_classes            
            probabily_dict[video_name[i]] = {'probabilities' : init_list,'target_label' : targets[i]}
            
                
       
        outaslist = outputs[i].tolist()
        probabily_dict[video_name[i]]['probabilities'] = list( map(add, probabily_dict[video_name[i]]['probabilities'], outaslist) )
            
            
            
            #print(probabily_dict[video_name[i]]['probabilities'])
            #with open("prob_print.txt",'a+') as f:
            #    f.write(video_name[i])
            #    f.write("-target:")
            #    f.write(str(targets[i].item()))
            #    f.write(",")
            #    f.write("{")
            #    for prob in probabily_dict[video_name[i]]['probabilities']:
            #        f.write(str(round(prob,2)))
            #        f.write(",")
            #    f.write("}")
            #    f.write("\n")
            


                
def calculate_accuracy_video_level(outputs, targets,video_name,results_dict):
    batch_size = outputs.size(0)
   
    for i in range(batch_size):
        _, pred = outputs[i].topk(1, 0, True)
        
        if video_name[i] not in results_dict:
            results_dict[video_name[i]] = []

        if pred.item() == targets[i].item():
            results_dict[video_name[i]].append(1)
        else:
            results_dict[video_name[i]].append(0)


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    #print("outputs",outputs.shape)
    #print("targets",targets)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size



