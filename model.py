import torch
import sys
#print(sys.path)
#sys.path.append('.')
from torch import nn
from mfnetwithmotionVectors import MFNET_MV
from models import  resnext, resnet
from mfnet_3d import MFNET_3D
from mfnet_3d_arch_change import MFNET_3D as MFNET_3D_C
from  mfnet_3d_multiple_exits_small import MFNET_3D as MFNET_3D_S
from  mfnet_3d_cbam import MFNET_3D as MFNET_3D_CBAM


#from mfnet_3d_multiple_exits imort MFNET_3D as MFNET_3D_MULT_LOSS
from transfer_model import MFNET_3D_T
from models import resnext
from torch import nn

def compress_dict(dict,opt):
    compressed_dict = {}
    for k, v in dict['state_dict'].items():
        if "num_batches_tracked" not in k:
            if v.shape[0] == 16:
                num_filters = int(v.shape[0])
            else:
                num_filters = int(v.shape[0]/2)
            if len(v.shape) > 1:
                if v.shape[1] == 3 or v.shape[1] ==16 or v.shape[1] ==1:
                    num_ch = int(v.shape[1])
                else:
                    num_ch = int(v.shape[1]/2)
        if "weight" in k and "bn" not in k:
            if "classifier" in k:
                compressed_dict[k] = v[:,:num_ch]
            else:
                compressed_dict[k] =  v[:num_filters,:num_ch,:,:,:]

        else:
            if "num_batches_tracked" not in k:
                compressed_dict[k] =  v[:num_filters]
            if "classifier" in k:
                print(k,v.shape)
                print(num_ch)
                compressed_dict[k] = v[:opt.n_classes]

    return compressed_dict
    
def generate_model(opt):
    assert opt.model in ['mfnet', 'resnext', 'resnet']
    opt.cbam = 0
    if opt.model == 'mfnet':
        if opt.mult_loss:
            model = MFNET_3D_T(opt)
        elif opt.time_focus:
            model = MFNET_3D_C(opt)
        #elif opt.mv:
        #    model = MFNET_MV(opt)
        elif opt.small:
            model = MFNET_3D_S(opt)
            pretrain = torch.load(opt.pretrain_path) 
            compressed_dict = compress_dict(pretrain,opt)
            model.cuda()
            model = nn.DataParallel(model)
            model.load_state_dict(compressed_dict,strict=True)
        
     
        else:
            model = MFNET_3D(opt.n_classes)
        
        
    elif opt.model == 'resnext':
        model = resnext.resnet101(opt)
    elif opt.model == 'resnet':
        model = resnet.resnet101(opt)
    if opt.mult_loss:
        param = model.student_model.parameters()
    else:
        param = model.parameters()
        

            
    return model, param
                

    
