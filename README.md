
# MFCD-Net: Action Recognition in the Compressed Domain
Code for submission of Neurips 2019 paper entitled "ARCDnet: Accelerating Action Recognition In The Compressed Domain"
Under contruction, Update every single day.

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
Offical Crawler:https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics
Another git if the first doesn't work:
https://github.com/Showmax/kinetics-downloader
1.video2jpeg using:
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
2.create a file that holds the number of frames:
python utils/n_frames_kinetics.py jpg_video_directory
3. If you download through the official crawler you can get the annotation using this command:
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path dst_json_path
other wise you can use the annotation file under "annotation_dir".

**Hmdb51:**
Download from:
http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
1.video2jpeg using:
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
2.create a file that holds the number of frames:
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
3. Annotation you use the ready annotation file in the "annotation_dir".

**Ucf101:**
Download from:
https://www.crcv.ucf.edu/data/UCF101.php
1.video2jpeg using:
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
2.create a file that holds the number of frames:
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
3. Annotation you use the ready annotation file in the "annotation_dir".


This git used the structure and base code of:https://github.com/kenshohara/3D-ResNets-PyTorch and the extraction of compressed component using this git:https://github.com/chaoyuaw/pytorch-coviar, so a big thanks to both of those projects, they did a great work.
