import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import torch
import copy
#from coviar import load, get_num_frames
from utils import load_value_file
import numpy.random as random
import cv2


#from transforms import color_aug


def pil_loader(path,type='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(type)

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
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
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
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations


def make_dataset(opt, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):


     
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    print(len(video_names))
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    
    dataset = []
    total_number_of_clips = 0
    gflop_per_clip = 8.53
    for i in range(len(video_names)):
    
        video_path =  video_names[i]
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        j_path  = os.path.join(opt.jpeg_path, video_names[i]) 
        n_frames_file_path = os.path.join(j_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))

        if n_frames <= 0:
            continue

        
        
        i_path = os.path.join(opt.iframe_path, video_names[i]) 
        
        m_path = os.path.join(opt.mv_path, video_names[i]) 
        r_path = os.path.join(opt.residual_path, video_names[i]) 
        #print("i_path:",i_path,"m_path:",m_path,"r_path:",r_path)
        #stop
        if not os.path.exists(i_path):
            print(i_path,"not exists")
            continue
        if not os.path.exists(r_path):   
            print(r_path,"not exists")
            continue
        
        begin_t = 1
        end_t = n_frames
        
        sample = {
            'video'    : video_path,
            'jpeg_path': j_path,
            'iframe_path': i_path,
            'mv_path': m_path,
            'residual_path': r_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1
        #dataset.append(sample)
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:###teesting
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                """
                if n_samples_for_each_video == 0:##means we are doing video level validation
                    rand = random.randint(2)
                    if rand == 1:
                        total_number_of_clips += 1
                        dataset.append(sample_j)
                else:
                """
                total_number_of_clips += 1
                dataset.append(sample_j)
                

    if n_samples_for_each_video == 0:
        num_of_videos = len(video_names)
        avg_clips_per_video = round(total_number_of_clips/num_of_videos,2)
        print("Number of videos:",num_of_videos)
        print("Number of clips:",total_number_of_clips)
        print("Avarage amount of clips per video:",avg_clips_per_video)
        print("Number of GFlos for each video in avarage will be,(true to mfnet only):",gflop_per_clip*avg_clips_per_video)
    return dataset, idx_to_class


class UCF101(data.Dataset):
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

    def __init__(self,
                 opt,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=12,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            opt, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        """
        with open("ucf101.txt",'w') as f:
            for item in  self.data:
                f.write("%s\n" % item)
        """
        
        self.opt = opt
        self.subset = subset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        jpg_dir_path = self.data[index]['jpeg_path']
        num_frames = self.data[index]['n_frames']
        gop_size = 12
        num_of_gops = int(num_frames/gop_size) #+ 1
        temp_num_of_gops = num_of_gops  + 1


        gop_index = int(self.data[index]['frame_indices'][0] / 12)
        #print(gop_index)
        iframe_path_img   = os.path.join(self.data[index]['iframe_path'],'iframe_') + str(gop_index) + '_' + str(0) + '.png'
        clip = []
        if not self.opt.residual_only:
            if os.path.exists(iframe_path_img):
                iframe            = pil_loader(iframe_path_img)
            else:
                gop_index -= 1
                iframe_path_img   = os.path.join(self.data[index]['iframe_path'],'iframe_') + str(gop_index) + '_' + str(0) + '.png'
                iframe            = pil_loader(iframe_path_img)
            clip.append(iframe)
        for frame_index in range(1,12):
            residual_path_img = os.path.join(self.data[index]['residual_path'],'residuals_') + str(gop_index) + '_' + str(frame_index) + '.png'
            #print("loading residual:",residual_path_img)
            if os.path.exists(residual_path_img):
                residual          = pil_loader(residual_path_img)
                if self.opt.residual_only and frame_index == 1:
                    clip.append(residual)
                clip.append(residual)

        
        
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            if len(clip) < 12:
                delta = 12 - len(clip)
                dup = clip[-1]
                for k in range(delta):
                    clip.append(dup)
            
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        if self.opt.add:
            l = 0
            iframe = clip[:,0,:,:]
            for mat in clip:
                if l != 0:
                    residual_frame = clip[:,l,:,:]
                    clip [:,l,:,:] = (clip[:,l,:,:] + iframe)/2
                l += 1


        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.opt.video_level_accuracy:
            return clip, target, self.data[index]['video'].split('/')[-1]
        else:
            return clip, target

    def __len__(self):
        return len(self.data)
