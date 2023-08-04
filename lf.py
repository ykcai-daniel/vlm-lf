import cv2
import os
import time
import json
import numpy as np
import datetime
import argparse
from test_cases import test_cases
from frame_processor import vlm_processor,visualize_results_lang,visualize_result_image
import torch
def process_video_owl_lang(video_path:str,text_queries,interval=6,result_dir=None,max_frame=None,result_video=None,run_type='lang'):
    print(f"Searching with Language Queries: {text_queries} in video {video_path}")
    device='cpu'
    all_frame_results=[]
    if torch.cuda.is_available():
        device='cuda'
    vlm_processor.model.to(device)
    print("Using "+device)
    video=cv2.VideoCapture(video_path)
    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        #however avc1 is not supported on linux if you install opencv-python with pip
        #in backend, consider converting video with ffmpeg
        #https://stackoverflow.com/questions/59290569/opencv-video-writer-unable-to-find-codec-or-avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
            result = remove_zero_boxes(result)
            result['boxes']=non_max_suppression_fast(np.array(result['boxes']), 0.3)
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

def process_video_owl_image(video_path:str,image_queries_names,interval=6,result_dir=None,max_frame=None,result_video=None,run_type='image'):
    print(f"Searching with image queries {image_queries_names} in video {video_path}")
    device='cpu'
    all_frame_results=[]
    if torch.cuda.is_available():
        device='cuda'
    vlm_processor.model.to(device)
    print ('Useing '+ device)
    image_queries=[cv2.imread(name) for name in image_queries_names]
    video=cv2.VideoCapture(video_path)
    if result_dir is not None:
        os.makedirs(f'./results/{result_dir}',exist_ok=True)
    if result_video is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer=cv2.VideoWriter(result_video,fourcc,round(30.0/interval),(int(video.get(3)),int(video.get(4))))
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
            result = remove_zero_boxes(result)
            result['boxes']=non_max_suppression_fast(np.array(result['boxes']), 0.3)
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
    os.makedirs('results',exist_ok=True)
    video_raw_name=video_name.split('/')[-1]
    str_max_frame=''
    if max_frame is not None:
        str_max_frame=f'_frame_{max_frame}'
    #append time stamp into name to avoid duplication
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M")
    result_video_name=f'./results/{video_raw_name}{str_max_frame}_{formatted_time}_{run_type}.mp4'
    result_json_name=f'./results/{video_raw_name}{str_max_frame}_{formatted_time}_{run_type}.json'
    print(f'Result Video Path: {result_video_name}')
    if run_type=='image':
        result=process_video_owl_image(video_name,queries,interval,result_dir=None,max_frame=max_frame,result_video=result_video_name)
        with open(result_json_name,'w') as f:
            print(f"Result Json: {result_json_name}")
            json.dump({
                'query':queries,
                'type':run_type,
                'result':result,
                
            },f)
    elif run_type=='lang':
        result=process_video_owl_lang(video_name,queries,interval,result_dir=None,max_frame=max_frame,result_video=result_video_name)
        with open(result_json_name,'w') as f:
            print(f"Result Json: {result_json_name}")
            json.dump({
                'query':queries,
                'type':run_type,
                'result':result,
                
            },f)
    else:
        print("Invalid run_type: must be image or lang ")

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        to_zero = np.where(overlap > overlapThresh)[0]
        x1[idxs[to_zero]] = 0
        y1[idxs[to_zero]] = 0
        x2[idxs[to_zero]] = 0
        y2[idxs[to_zero]] = 0
        idxs = np.delete(idxs, last)
    return boxes.astype("int").tolist()

def remove_zero_boxes(result):
    if 'boxes' not in result:
        return result
    new_result={k:[] for k in result}
    for index,box in enumerate(result['boxes']):
        if box != [0, 0, 0, 0]:
            for k in result:
                new_result[k].append(result[k][index])
    return new_result

    
#If you are using CUHK CSE slurm cluster
#export SLURM_CONF=/opt1/slurm/gpu-slurm.conf 
#srun --gres=gpu:1 -w gpu39 --pty /bin/bash
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--video_name',type=str)
    parser.add_argument('--query_index',type=int)
    parser.add_argument('--max_frame',type=int,default=None)
    args=parser.parse_args()
    video_name=args.video_name
    query=test_cases[args.query_index]['object']
    query_type=test_cases[args.query_index]['type']
    run_video_to_video(video_name,query,query_type)


    
