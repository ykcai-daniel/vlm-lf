from transformers import pipeline
import skimage
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2 
import os
import argparse
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

class FrameProcessor:
    def __init__(self) -> None:
        checkpoint = "google/owlvit-base-patch32"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def process_image(self,image,text_queries):
        inputs=self.processor(text=text_queries, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes=[(image.shape[0],image.shape[1])]
            raw_results = self.processor.post_process_object_detection(outputs, threshold=0.1,target_sizes=target_sizes)[0]
        results={
            'scores':raw_results["scores"].tolist(),
            'labels':raw_results["labels"].tolist(),
            'boxes':raw_results["boxes"].tolist()
        }
        return results

def visualize_results(image,result,classes):
    #different color for different bounding box
    colors=[(255-round(255/i),round(255/i)) for i in range(len(classes)+1,1,-1)]
    class_string=", ".join([f"{index+1}->{c}" for index,c in enumerate(classes)])
    format_string=f"Classes: [{class_string}]"
    cv2.putText(image, format_string,(5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    for index,label in enumerate(result['labels']):
        point=result['boxes'][index]
        p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
        #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
        cv2.rectangle(img=image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=colors[label],thickness=3)
        cv2.putText(image, f"{label+1}",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image


vlm_processor=FrameProcessor()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--img",type=str,default='./data/SmartCity/images/39.jpg')
    args = parser.parse_args()
    checkpoint = "google/owlvit-base-patch32"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)
    image = cv2.imread(args.img)
    print(f"Image shape: {image.shape}")
    #text_queries=["human face", "rocket", "nasa badge", "star-spangled banner"]
    text_queries=['backpack','lift','girl in pink']
    inputs=processor(text=text_queries, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes=[(image.shape[0],image.shape[1])]
        raw_results = processor.post_process_object_detection(outputs, threshold=0.1,target_sizes=target_sizes)[0]
    results={
        'scores':raw_results["scores"].tolist(),
        'labels':raw_results["labels"].tolist(),
        'boxes':raw_results["boxes"].tolist()
    }
    visualize_image=visualize_results(image,results,text_queries)
    cv2.imwrite(f"{str(os.path.basename(os.path.join(args.img)))}_process.jpg",visualize_image)