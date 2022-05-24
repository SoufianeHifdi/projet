#! /bin/bash

# Since conda is a bash function and bash functions can not be propagated to independent shells (e.g. opened by executing a bash script), one has to add the line
source /home/sfifina/miniconda3/etc/profile.d/conda.sh


# Execution du tracking
cd yolov4-deepsort
conda activate yolov4-cpu
python3 object_tracker.py --video ./videos/video_cam1.mp4 --model yolov4
python3 object_tracker.py --video ./videos/video_cam2.mp4 --model yolov4
python3 object_tracker.py --video ./videos/video_cam3.mp4 --model yolov4
python3 object_tracker.py --video ./videos/video_cam4.mp4 --model yolov4

# Pour utiliser tinyYolo : 
# python3 object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./videos/video_cam3.mp4 --tiny 
# python3 object_tracker.py --weights ./checkpoints/yolov4-tiny 416 --model yolov4 --video ./data/video/test.mp4 --tiny 

#Partie Execution du Reid
cd ~/Pres/deep-person-reid
conda activate torchreid 
python3 object_reid.py

exit 0
