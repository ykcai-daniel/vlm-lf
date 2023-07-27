import cv2
import os
from frame_processor import vlm_processor,visualize_results
from clip_processor import Clip
import json
def process_video_owl(video_path:str,text_queries,interval=6,result_dir=None):
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
        result=vlm_processor.process_image(frame,text_queries)
        print(f"Results of visualize_results {frame_count}: {result['boxes']}")
        visualized_image=visualize_results(frame,result,text_queries)
        result_path=f'./results/{result_dir}/frame_{frame_count}.jpg'
        cv2.imwrite(result_path,visualized_image)
    video.release()
def process_video_clip(model:Clip,video_path:str,text_queries,interval=6,max_frame=None):
    result={s:[] for s in text_queries}
    video=cv2.VideoCapture(video_path)
    frame_count=0
    print(f"Number of frames: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} Processing interval: {interval}")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Ret is false!")
            break
        if max_frame is not None:
            if frame_count>max_frame:
                print(f"Max frame {max_frame} reached.")
                video.release()
                return result
        frame_count=frame_count+1
        if frame_count%interval!=0:
            continue
        logits=model.get_logits(frame,text_queries)
        print(f"Results of frame {frame_count}: {logits}")
        for index in range(len(text_queries)):
            result[text_queries[index]].append(logits[0][index].item())
    video.release()
    return result
#If you are using CUHK CSE slurm cluster
#export SLURM_CONF=/opt1/slurm/gpu-slurm.conf 
#srun --gres=gpu:1 -w gpu30 --pty /bin/bash
if __name__=='__main__':
    video_name='./data/hong_kong_airport_3.mp4'
    text_queries=[
        'white backpack','white suitcase','blue backpack','blue suitcase'
    ]
    clip_queries=[f'an image with {t}' for t in text_queries]
    clip_queries.append('an image of crows at an airport')
    # text_queries=[
    #     'checkered tote',
    # ]
    # use owl-vlm
    #process_video_owl(video_name,text_queries,result_dir='hong_kong_airport_3')
    #use clio
    clip=Clip()
    res=process_video_clip(clip,video_name,clip_queries,max_frame=450)
    with open('hong_kong_airport_3.json','w') as f:
        json.dump(res,f)
