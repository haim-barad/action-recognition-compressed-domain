


# MFCD-Net: Action Recognition in the Compressed Domain
Code for submission of Neurips 2019 paper entitled "MFCD-Net: Accelerating Action Recognition In The Compressed Domain"

## Under contruction, Update every single day.

<h2>Requirements (further instruction will be added)</h2>
<ol>
<li>python3
<li>pytorch1.0
<li>torchvision
<li>pillow
<li>ffmpeg
<li>opencv:<br/></ol>

>  pip install opencv-python 


if the **use** of opencv returning an error, try:
>apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev

**Coviar:**   

Install using the instruction below:
https://github.com/chaoyuaw/pytorch-coviar/blob/master/GETTING_STARTED.md

comments:
 - If the make clean doesn't work, continue, don't give up.
 - If you get errors in the last command, try install those libraries:

        apt-get update 
        apt-get install libbz2-dev 
        apt-get install -y liblzma-dev 
        apt-get install libavutil-dev
        apt-get install libavcodec-dev 
        apt-get install libavformat-dev
        apt-get install libswscale-dev

</ol>


## Datasets:
**Kinetics400:**
Download from the offical Crawler:
https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics
Another git if the first doesn't work:
https://github.com/Showmax/kinetics-downloader

 1. video2jpeg using:
	  >python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
 2. create a file that holds the number of frames:
	  >python utils/n_frames_kinetics.py jpg_video_directory
 3.  If you download through the official crawler you can get the annotation using this   command
	 >python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path  dst_json_path

     otherwise you can use the annotation file under "annotation_dir".

**Hmdb51:**
Download from:
http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

 1. video2jpeg using:
	  > python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory.
 
 2. create a file that holds the number of frames:
	  > python utils/n_frames_ucf101_hmdb51.py jpg_video_directory.
 
 3. Annotation you use the ready annotation file in the "annotation_dir".




**Ucf101:**
Download from:
https://www.crcv.ucf.edu/data/UCF101.php

 1. Video2jpeg using:
     >python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
 2. Create a file that holds the number of frames:
	>python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
 3. Annotation you use the ready annotation file in the "annotation_dir".
## Extractign Compressed Representation Data


## Run
### Train
**Final goal**: having MFCD-Net running in the compressed domain on HMDB51 datatset

 1. Download weights of MultiFiberNet model pretrain on kinetics from:[https://github.com/cypw/PyTorch-MFNet](https://github.com/cypw/PyTorch-MFNet)
 2. Finetune the model on hmdb51 on the uncompressed domain , while using 12 frames and not 16 as the original model.
 

> python main.py  --model mfnet --sample_size 224 --sample_duration 12 
> --batch_size <batch_size> --jpeg_path <jpeg_path>  --mfnet_st --annotation_path annotation_dir/hmdb51/hmdb51_1.json --pretrain_path  path/MFNet3D_Kinetics-400_72.8.pth --dataset hmdb51 --n_classes 51 --learning_rate 0.005 --weight_decay 0.001 --result_path standard_12frames_results_hmdb51
 --n_epochs <number_of_epochs>

3.	Finetune the model on hmdb51 on the compressed domain 

>  python main.py  --model mfnet --sample_size 224 --sample_duration 12 --batch_size <batch_size> --jpeg_path <jpeg_path>  --mfnet_st --annotation_path annotation_dir/hmdb51/hmdb51_1.json --pretrain_path standard_12frames_results/<weights_name>.pth  --dataset hmdb51 --n_classes 51 --learning_rate 0.005 --weight_decay 0.001 --result_path mfcdnet_results_hmdb51 --n_epochs <number_of_epochs> --compressed --iframe_path <iframe_directory_path> --residual_path <residual_directory_path> 

 
 ## Refernces
This git used the structure and base code of:https://github.com/kenshohara/3D-ResNets-PyTorch and the extraction of compressed component using this git:https://github.com/chaoyuaw/pytorch-coviar,
A big thanks to both of those projects, they did a great work and their citation information is below.

Citation of the paper which we used their gits:
```
@inproceedings{wu2018coviar,
  title={Compressed Video Action Recognition},
  author={Wu, Chao-Yuan and Zaheer, Manzil and Hu, Hexiang and Manmatha, R and Smola, Alexander J and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={CVPR},
  year={2018}
}
```
and
```
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```

