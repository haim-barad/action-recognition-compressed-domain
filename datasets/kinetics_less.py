import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import numpy  as np
import functools
import json
import copy
from torchvision import transforms

from coviar import load
from coviar import get_num_frames

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        #print("image_path is :",image_path)
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
         #   print("image_path doesnot exist :",image_path)
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(opt,root_path, annotation_path, subset, n_samples_for_each_video,
                 frames_sequence):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print(len(video_names))
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        #print(video_names[i])
        if subset == "training":
            video_path = os.path.join(opt.video_path, video_names[i]) + ".mp4"
            if not os.path.exists(video_path):
                continue
                #root_path = "/dataset1/kinetics2_jpg/train/"
                #video_path = os.path.join(root_path, video_names[i])
                #if not os.path.exists(video_path):
                
        elif subset == "validation":
            root_path="/kinetics2/kinetics2/kinetics_val_reencode"
            video_path = os.path.join(root_path, video_names[i])+ ".mp4"
            #print(video_path)
            if not os.path.exists(video_path):
                continue

        
        n_frames = get_num_frames(video_path)
        if n_frames <= 0:
            continue

        sample = {
            'video': video_path,
            'n_frames': n_frames,
            'video_id': video_names[i]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - frames_sequence) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = frames_sequence
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + frames_sequence)))
                #print (sample_j['frame_indices'])
                if len(sample_j['frame_indices']) == 12:
                    dataset.append(sample_j)
    print(len(dataset))
    return dataset, idx_to_class
    
def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

class Kinetics(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,opt,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 frames_sequence=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(opt,
            root_path, annotation_path, subset, n_samples_for_each_video,
            frames_sequence)
        print(len(self.data))
        self.opt = opt
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.tensor_2_image = transforms.ToPILImage()
        self.totensor = transforms.ToTensor()
        self.subset = subset
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        #if self.temporal_transform is not None:
        #    frame_indices = self.temporal_transform(frame_indices)
        clip = []
        mvclip = [] 
        #print(path,":",int(frame_indices[0] / 12))
        gop_index = int(frame_indices[0] / 12)
        if not self.opt.residual_only:
           #iframe_path_img   = os.path.join(self.data[index]['iframe_path'],'iframe_') + str(gop_index) + '_' + str(0) + '.png'
            iframe            = load(path,gop_index,0, 0, True)
            iframe =self.tensor_2_image(iframe)
            #print("iframe shape b44:",iframe.shape)
            #iframe = iframe.reshape(iframe.shape[2],iframe.shape[1],iframe.shape[0])
            #print("iframe shape after:",iframe.shape)
            clip.append(iframe)
        for frame_index in range(1,12):
            residual          = load(path,gop_index,frame_index, 2, True)
            residual += 128
            residual = (np.minimum(np.maximum(residual, 0), 255)).astype(np.uint8)
            #residual = residual.reshape(residual.shape[2],residual.shape[1],residual.shape[0])
            residual = self.tensor_2_image(residual)
            clip.append(residual)

            #print("residual shape b4:",residual.shape)
            #residual = residual.reshape(residual.shape[2],residual.shape[1],residual.shape[0])
            #print("residual shape after:",residual.shape)
            
            if self.opt.residual_only and frame_index == 1: ## double if we skip the iframe
                clip.append(residual)
            """
            if self.opt.mv:
                mv_path_img       = os.path.join(self.data[index]['mv_path'],'motionvectors_') + str(gop_index) + '_' + str(frame_index) + '.png'
                if os.path.exists(mv_path_img):
                    mv          = pil_loader(mv_path_img,'LA')
                    if frame_index == 1: ##feeding twice intentually
                        mvclip.append(mv)
                    mvclip.append(mv)
                else:
                    continue
            """
        p = 0
        new_clip = []
        if self.spatial_transform is not None:
            
            self.spatial_transform.randomize_parameters()
            
               
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.opt.video_level_accuracy and self.subset == 'validation':
            return clip, target, self.data[index]['video'].split('/')[-1]
        else:            
            return clip, target

    def __len__(self):
        return len(self.data)
