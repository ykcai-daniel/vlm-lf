import cv2
import os
from frame_processor import vlm_processor,visualize_results

def process_video(video_path:str,text_queries,interval=6,result_dir=None):
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

#If you are using CUHK CSE slurm cluster
#export SLURM_CONF=/opt1/slurm/gpu-slurm.conf 
#srun --gres=gpu:1 -w gpu30 --pty /bin/bash
if __name__=='__main__':
    video_name='./data/hong_kong_airport_3.mp4'
    # text_queries=[
    #     'white backpack',
    #     'black backpack',
    #     'white suitcase',
    #     'black suitcase',
    # ]
    text_queries=[
        'checkered tote',
    ]
    process_video(video_name,text_queries,result_dir='hong_kong_airport_3')
