import matplotlib.pyplot as plt
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

    def from_data_dict(self,data):
        self.query_type=data['type']
        self.labels=data['query']
        for frame_result in data['result']:
            new_frame_result=FrameResult()
            new_frame_result.from_data_dict(frame_result)
            self.frame_results.append(new_frame_result)

    
    #return a dict: label to frame number sorted by logit field
    #only includes non empty frames
    def get_bbox_logits(self):
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

    def sort_logits_frame_max(self,top_k=None,default=-100):
        def max_scorer(x):
            if len(x)==0:
                return default
            return max(x)
        return self.sort_logits_frame(max_scorer,top_k)

    def sort_logits_frame(self,frame_scorer,top_k=None):
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

import json
if __name__=='__main__':
    #lang query
    #change to the name of the json output by lf.py
    result_json_file='results/hong_kong_airport_demo_data.mp4_202308041633_image.json'
    video_results=VideoResult()
    with open(result_json_file) as f:
        json_data=json.load(f)
        video_results.from_data_dict(json_data)
    print(f"Video results loaded")
    print(video_results.sort_logits_frame_max(top_k=50))
    # bbox_logit=video_results.get_bbox_logits()
    # print(bbox_logit)
    # print(f"Data length: {len(bbox_logit)} labels: {list(bbox_logit.keys())}")
    # visualize_logits(bbox_logit)
    