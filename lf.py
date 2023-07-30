import cv2
import os
import time
import json
from frame_processor import vlm_processor,visualize_results_lang,visualize_result_image
import torch
def process_video_owl_lang(video_path:str,text_queries,interval=6,result_dir=None,max_frame=None,result_video=None):
    print(f"Searching with Language Queries: {text_queries} in video {video_path}")
    device='cpu'
    all_frame_results=[]
    if torch.cuda.is_available():
        print("Using cuda")
        device='cuda'
        vlm_processor.model.to(device)
    video=cv2.VideoCapture(video_path)
    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        #however avc1 is not supported on linux if you install opencv-python with pip
        #in backend, consider converting video with ffmpeg
        #https://stackoverflow.com/questions/59290569/opencv-video-writer-unable-to-find-codec-or-avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fps is 5
        video_writer=cv2.VideoWriter(result_video,fourcc,round(30.0/interval),(int(video.get(3)),int(video.get(4))))
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if max_frame is not None:
            if max_frame<=frame_count:
                break
        if frame_count%interval==0:
            start=time.perf_counter()
            result=vlm_processor.process_image(frame,text_queries,device)
            result['frame']=frame_count
            all_frame_results.append(result)
            end=time.perf_counter()
            print(f"Results of frame {frame_count}: {result['scores']} Time {end-start}s")
            visualized_image=visualize_results_lang(frame,result,text_queries)
            if result_dir is not None:
                result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
                cv2.imwrite(result_path,visualized_image)
            if result_video is not None:
                video_writer.write(visualized_image)
        else:
            all_frame_results.append({'frame':frame_count})
        frame_count=frame_count+1
    video_writer.release()
    video.release()
    return all_frame_results

def process_video_owl_image(video_path:str,image_queries_names,interval=6,result_dir=None,max_frame=None,result_video=None):
    print(f"Searching with image queries {video_path} in video {video_path}")
    device='cpu'
    all_frame_results=[]
    if torch.cuda.is_available():
        print("Using cuda")
        device='cuda'
    vlm_processor.model.to(device)
    image_queries=[cv2.imread(name) for name in image_queries_names]
    video=cv2.VideoCapture(video_path)
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fps is 5
        video_writer=cv2.VideoWriter(result_video,fourcc,round(30.0/interval),(int(video.get(3)),int(video.get(4))))
        print(video_writer)
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        if max_frame is not None:
            if max_frame<=frame_count:
                break
        if frame_count%interval==0:
            start=time.perf_counter()
            result=vlm_processor.image_query(frame,image_queries,device)
            result['frame']=frame_count
            all_frame_results.append(result)
            end=time.perf_counter()
            print(f"Results of frame {frame_count}: {result['scores']} Time:{end-start}s")
            visualized_image=visualize_result_image(frame,result,f"Image: {image_queries_names[0]}")
            if result_dir is not None:
                result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
                cv2.imwrite(result_path,visualized_image)
            if result_video is not None:
                video_writer.write(visualized_image)
        else:
            all_frame_results.append({'frame':frame_count})
        frame_count=frame_count+1
    video_writer.release()
    video.release()
    return all_frame_results

def run_video_to_video(video_name:str,queries:list[str],run_type:str,interval=3,max_frame=None):
    video_raw_name=video_name.split('/')[-1]
    str_max_frame=''
    if max_frame is not None:
        str_max_frame=f'_frame_{max_frame}'
    if run_type=='image':
        result=process_video_owl_image(video_name,queries,interval,result_dir=None,max_frame=max_frame,result_video=f'./results/{video_raw_name}_{str_max_frame}_img.mp4')
        with open(f'./results/{video_raw_name}{str_max_frame}_img.json','w') as f:
            json.dump({
                'query':queries,
                'result':result,
            },f)
    elif run_type=='lang':
        result=process_video_owl_lang(video_name,queries,interval,result_dir=None,max_frame=max_frame,result_video=f'./results/{video_raw_name}_{str_max_frame}_lang.mp4')
        with open(f'./results/{video_raw_name}{str_max_frame}_lang.json','w') as f:
            json.dump({
                'query':queries,
                'result':result,
            },f)
    else:
        print("Invalid run_type: must be image or lang ")

    
#If you are using CUHK CSE slurm cluster
#export SLURM_CONF=/opt1/slurm/gpu-slurm.conf 
#srun --gres=gpu:1 -w gpu39 --pty /bin/bash
if __name__=='__main__':
    video_name='data/hong_kong_airport_demo_data.mp4'
    text_queries=[
        'white backpack','white suitcase','black backpack','black suitcase','black and white striped backpack','tote bag'
    ]
    image_queries=['test_images/pink_short_luggage.jpg']
    #run with language
    run_video_to_video(video_name,text_queries,'lang')
    #or run with imange 
    #run_video_to_video(video_name,image_queries,'image')

    
