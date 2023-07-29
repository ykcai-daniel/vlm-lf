import cv2
import os
from frame_processor import vlm_processor,visualize_results_lang,visualize_result_image
from clip_processor import Clip
import json
import torch
def process_video_owl_lang(video_path:str,text_queries,interval=6,result_dir=None):
    device='cpu'
    if torch.cuda.is_available():
        print("Using cuda")
        device='cuda'
        vlm_processor.model.to(device)
    video=cv2.VideoCapture(video_path)
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_count=frame_count+1
        if frame_count%interval!=0:
            continue
        result=vlm_processor.process_image(frame,text_queries,device)
        print(f"Results of frame {frame_count}: {result['scores']}")
        if result_dir is not None:
            visualized_image=visualize_results_lang(frame,result,text_queries)
            result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
            cv2.imwrite(result_path,visualized_image)
    video.release()

def process_video_owl_image(video_path:str,image_queries_names,interval=6,result_dir=None,max_frame=None):
    device='cpu'
    if torch.cuda.is_available():
        print("Using cuda")
        device='cuda'
    vlm_processor.model.to(device)
    image_queries=[cv2.imread(name) for name in image_queries_names]
    video=cv2.VideoCapture(video_path)
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_count=frame_count+1
        if frame_count%interval!=0:
            continue
        result=vlm_processor.image_query(frame,image_queries,device)
        print(f"Results of frame {frame_count}: {result['scores']}")
        if result_dir is not None:
            visualized_image=visualize_result_image(frame,result,f"Image: {image_queries_names[0]}")
            result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
            cv2.imwrite(result_path,visualized_image)
    video.release()
    
#If you are using CUHK CSE slurm cluster
#export SLURM_CONF=/opt1/slurm/gpu-slurm.conf 
#srun --gres=gpu:1 -w gpu39 --pty /bin/bash
if __name__=='__main__':
    video_name='data/hong_kong_airport_demo_data.mp4'
    text_queries=[
        'white backpack','white suitcase','black backpack','black suitcase'
    ]
    image_queries=['test_images/pink_short_luggage.jpg']
    process_video_owl_lang(video_name,text_queries,result_dir='hong_kong_airport_demo_text')
    
