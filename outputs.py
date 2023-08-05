import matplotlib.pyplot as plt
import cv2
import os
import torchvision
from PIL import Image
from utils.utils import file_name,split_suffix
prop_loc={
        'scores':0,
        'logits':1,
        'boxes':2,
        'labels':3,
    }
class FrameResult:
    
    def __init__(self) -> None:
        #data: [score, logit,bbox,label] if query is image then all labels are 0.
        self.data=None
        self.frame_index=None


    def from_data_dict(self,data):
        self.data=[]
        self.frame_index=data['frame']
        if 'scores' in data:
            for i,_ in enumerate(data['scores']):
                self.data.append([data['scores'][i],data['logits'][i],data['boxes'][i],data['labels'][i]])
        
    def to_dict(self):
        raise NotImplemented

    def get_prop(self,prop:str):
        return [record[prop_loc[prop]] for record in self.data]
    
    def get_result_by_label(self,label:int):
        return [record for record in self.data if record[3]==label] 

    def skipped(self):
        return self.data==None
    

    def nms(self,thresh):
        pass

    def remove_zero_box(self):
        pass

class VideoResult:

    def __init__(self) -> None:
        self.frame_results:list[FrameResult]=[]
        self.labels:list[str]=[]
        self.query_type=None

    #create the object based on the json generated by lf.py
    def from_data_dict(self,data):
        self.query_type=data['type']
        self.labels=data['query']
        for frame_result in data['result']:
            new_frame_result=FrameResult()
            new_frame_result.from_data_dict(frame_result)
            self.frame_results.append(new_frame_result)

    
    #return a dict: label to frame number sorted by logit field
    #only includes non empty frames
    def get_bbox_logits(self)->dict[str,list[int]]:
        results={}
        non_empty_frames=[fr.frame_index for fr in self.frame_results if not fr.skipped()]
        for label in self.labels:
            results[label]=[]
        for f in non_empty_frames:
            f_labels=self.frame_results[f].get_prop('labels')
            f_logits=self.frame_results[f].get_prop('logits')
            for index,l in enumerate(f_labels):
                results[self.labels[l]].append(f_logits[index])
        return results

    def sort_bbox_logits(self):
        raise NotImplementedError

    #return top k frames of each label in self.labels; 
    #use the max logit of each frame as the score of the frame
    def sort_logits_frame_max(self,top_k=None,default=-100)->dict[str,list[int]]:
        def max_scorer(x):
            if len(x)==0:
                return default
            return max(x)
        return self.sort_logits_frame(max_scorer,top_k)
    #return top k frames of each label in self.labels
    
    def sort_logits_frame(self,frame_scorer,top_k=None)->dict[str,list[int]]:
        sorted_frames={}
        non_empty_frames=[fr.frame_index for fr in self.frame_results if not fr.skipped()]
        for label in self.labels:
            def key_func(x,label):
                frame_logits=self.frame_results[x].get_prop('logits')
                frame_labels=self.frame_results[x].get_prop('labels')
                same_label_logits=[frame_logits[i] for i,l in enumerate(frame_labels) if self.labels[l]==label]
                return frame_scorer(same_label_logits)
            sorted_frames[label]=sorted(non_empty_frames,key=lambda x:key_func(x,label),reverse=True)
        if top_k is not None:
            for k in sorted_frames:
                sorted_frames[k]=sorted_frames[k][:top_k]
        return sorted_frames

    def get_frame_result(self,index):
        return self.frame_results[index]
    
    #TODO: write the frames in single pass
    def dump_top_k_frames(self,top_k,video_path:str):
        video_name=file_name(video_path)
        top_k_frames=self.sort_logits_frame_max(top_k)
        for label in top_k_frames:
            connected_name=label.replace(' ','_')
            image_name=file_name(connected_name)
            raw_name,suffix=split_suffix(image_name)
            video=cv2.VideoCapture(video_path)
            result_save_path=f'./results/{video_name}_{raw_name}'
            os.makedirs(result_save_path,exist_ok=True)
            print(f'Writing {len(top_k_frames[label])} image for {label}')
            print(f"Top-{top_k} frames saved to: {result_save_path}")
            for i,current_index in enumerate(top_k_frames[label]):
                #random access in video is actually slow
                video.set(cv2.CAP_PROP_POS_FRAMES, current_index)
                ret, frame = video.read()
                if not ret:
                    print(f"Error: Unable to read frame {current_index}")
                    continue
                write_path=f'./results/{video_name}_{raw_name}/{i}.jpg'
                label_location=self.labels.index(label)
                box_filtered=self.get_frame_result(current_index).get_result_by_label(label_location)
                for box in box_filtered:
                    point=box[2]
                    p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
                    cv2.rectangle(img=frame,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=(0,0,255),thickness=3)
                    cv2.putText(frame, "{:.2f}".format(box[1]),(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame, f"Frame {current_index} Rank {i}",(25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                cv2.imwrite(write_path,frame)
            video.release()



def visualize_logits(label_to_logits):
    for k in label_to_logits:
        print(f"bboxs count of {k}: {len(label_to_logits[k])}")
        plt.hist(label_to_logits[k], bins=50, color='b', edgecolor='black')
        plt.title(f'Scores Distribution of {len(label_to_logits[k])} {k}(s) ')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        hist_name=f"{k}_hist.jpg"
        print(f"Creating histogram {hist_name}")
        plt.savefig(hist_name)
        plt.clf()

def make_grid(image_dir,row_size=4):
    transform =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize(300)])
    images=[ transform(Image.open(os.path.join(image_dir,f'{i+1}.jpg'))) for i in range(row_size*row_size)]
    grid = torchvision.utils.make_grid(images, nrow=row_size,pad_value=4)
    torchvision.transforms.ToPILImage()(grid).save(f'{image_dir}/summary_{row_size}x{row_size}.jpg')


import json
if __name__=='__main__':
    #lang query
    #change to the name of the json output by lf.py
    result_json_file='results/hong_kong_airport_demo_data.mp4_202308041624_image.json'
    video_results=VideoResult()
    with open(result_json_file) as f:
        json_data=json.load(f)
        video_results.from_data_dict(json_data)
    print(f"Video results loaded")
    print(video_results.sort_logits_frame_max(top_k=50))
    video_results.dump_top_k_frames(50,'data/hong_kong_airport_demo_data.mp4')
    #change the location to where you want the grid
    make_grid('./results/hong_kong_airport_demo_data.mp4_pink_short_luggage')

    