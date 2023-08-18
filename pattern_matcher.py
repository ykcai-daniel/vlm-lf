import numpy as np
import time
from outputs import FrameResult
import json
import cv2
#TODO: match by class
#TODO: weight each IOU by frame score


#inputs:
#bbox in (x_0,y_0,x_1,y_1) format 
#frame_boxes: (len(frame),4)
#transformed_boxes: (len(pattern),4,len(parem))
#returns: (len(pattern),len(frame),len(parem)) i.e. an iou matrix for each parem combination
#reference: https://stackoverflow.com/questions/57897578/efficient-way-to-calculate-all-ious-of-two-lists
def _iou(frame_boxes,transformed_boxes):
    low = np.s_[:,:,(0,1),:]
    high = np.s_[:,:,(2,3),:]
    start=time.perf_counter()
    frame_boxes=np.expand_dims(frame_boxes,(0,3))
    transformed_boxes=np.expand_dims(transformed_boxes,(1))
    A,B = frame_boxes.copy(),transformed_boxes.copy()
    A[high] += 1 
    B[high] += 1
    #make sure the boxes are in xyxy format
    assert(np.all(A[high]>A[low]))
    assert(np.all(B[high]>B[low]))
    intersection_area = np.maximum(0,np.minimum(A[high],B[high])-np.maximum(A[low],B[low])).prod(2)
    a_area=(A[high]-A[low]).prod(2)
    b_area=(B[high]-B[low]).prod(2)
    union_area=a_area+b_area-intersection_area
    ious=intersection_area / union_area
    print(f"IOU max value: {np.max(ious)} median: {np.median(ious)}")
    #sanity check: iou<1
    assert(np.all(ious<=1))
    end=time.perf_counter()
    print(f"Time: {end-start}s")
    return ious

class ParemGrid:
    def __init__(self,frame_w=1280,frame_h=720,sampling_rate=50) -> None:
        parem_grid=np.mgrid[0:frame_w:sampling_rate,0:frame_h:sampling_rate,0.125:1.5:0.125]
        self.parem_grid=np.reshape(parem_grid,(3,-1),'F') #F stands for Fortran style row major ordering
    def xy(self):
        return self.parem_grid[(0,1),:]
    def k(self):
        return self.parem_grid[(2),:]

    def __len__(self):
        return self.parem_grid.shape[1]

class Pattern:
    
    def __init__(self,patter_list) -> None:
        self.data=patter_list
        self._np=np.array(patter_list)
        self.len=self._np.shape[0]

    def xywh(self):
        return self._np[:,[0,1,2,3]]
    def xyxy(self):
        return FrameResultNp.xywh2xyxy(self.xywh())

    def classes(self):
        self._np[:,4]

    #to preserve the shape, we must transform with xyxy format
    def transform_with_parem(self,parem:ParemGrid):
        xyxy=self.xyxy()
        lower_point=np.expand_dims(xyxy[:,(0,1)],(2))
        higher_point=np.expand_dims(xyxy[:,(2,3)],(2))
        k=np.expand_dims(parem.k(),(0,1))
        x_0_y_0=np.expand_dims(parem.xy(),(0))
        lower_scaled,higher_scaled=k*lower_point,k*higher_point
        lower_shifted=lower_scaled+x_0_y_0
        high_shifted=higher_scaled+x_0_y_0
        new_boxes=np.concatenate([lower_shifted,high_shifted],(1))
        return new_boxes
    
    def __len__(self):
        return self.len

    def visualize(self):
        image=cv2.imread('misc/frame_550.jpg')
        pattern_box_xyxy=FrameResultNp.xywh2xyxy(self._np).tolist()
        #different color for different bounding box
        for index,point in enumerate(pattern_box_xyxy):
            p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
            #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
            cv2.rectangle(image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=(0,255,0),thickness=3)
            cv2.putText(image, f"{index}",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite("pattern_visualized.jpg",image)

class FrameResultNp:
    @staticmethod
    def xyxy2xywh(xyxy):
        w=xyxy[:,2]-xyxy[:,0]
        w=np.expand_dims(w,axis=1)
        h=xyxy[:,3]-xyxy[:,1]
        h=np.expand_dims(h,axis=1)
        return np.concatenate([xyxy[:,(0,1)],w,h],axis=1)
    @staticmethod
    def xywh2xyxy(xywh):
        x=xywh[:,0]+xywh[:,2]
        x=np.expand_dims(x,axis=1)
        y=xywh[:,1]+xywh[:,3]
        y=np.expand_dims(y,axis=1)
        return np.concatenate([xywh[:,(0,1)],x,y],axis=1)
    
    def __init__(self,frame_result:FrameResult):
        self.data=frame_result.data
        numpy_arr=[]
        for record in self.data:
            score,logit,box,label=record
            numpy_arr.append([score,logit,*box,label])
        self._np=np.array(numpy_arr)
    def logits(self):
        pass
    def scores(self):
        pass

    def boxes_xywh(self):
        xyxy=self.boxes_xyxy()
        return self.xyxy2xywh(xyxy)
    def boxes_xyxy(self):
        return self._np[:,2:6]

def match_box(ious_of_parem):
    return np.argmax(ious_of_parem,axis=1)

def match_pattern_impl(frame_result_boxes,transformed_parem_boxes):
    #iou result: len(pattern)*len(frame)*len(parem)
    iou_frame_pattern=_iou(frame_result_boxes,transformed_parem_boxes)
    #use scipy linear assign to decide optimal box assignment for each parem group
    max_index=match_box(iou_frame_pattern)
    index_other_dim_1=np.tile(np.arange(max_index.shape[0]),(max_index.shape[1],1)).T
    index_other_dim_2=np.tile(np.arange(max_index.shape[1]),(max_index.shape[0],1))
    max_box_seleced=iou_frame_pattern[index_other_dim_1,max_index,index_other_dim_2]
    score_parem=np.average(max_box_seleced,axis=0)
    # print(f"Scores of parem: {score_parem} shape: {score_parem.shape}")
    max_loc=np.argmax(score_parem)
    print(f"Max loc:{max_loc} Max Score:{score_parem[max_loc]}")
    return max_loc

if __name__=="__main__":
    
    #use frame 550 of 100s video as test
    parem_grid=ParemGrid()
    #Person with two luggage pattern xywh
    pattern=Pattern([
        (0,100,50,100), #luggage one
        (50,0,75,200), #person
        (75+50,100,50,100), #luggage two
    ])
    frame_result=FrameResult()
    with open('misc/pattern_test.json') as f:
        data=json.load(f)
        frame_result.from_data_dict(data)
    frame_result=FrameResultNp(frame_result)
    #size: len(pattern)*(x,y,w,h)*len(parem) 
    start=time.perf_counter()
    transformed_boxes=pattern.transform_with_parem(parem_grid)
    max_parem_loc=match_pattern_impl(frame_result.boxes_xyxy(),transformed_boxes)
    print(f"Max Parem Loc: {max_parem_loc}")
    print(f"Max Parems: {parem_grid.parem_grid[:,max_parem_loc]}")
    best_boxes=transformed_boxes[:,:,max_parem_loc].tolist()
    print(f"Best Transformed Boxes: {best_boxes}")
    end=time.perf_counter()
    image=cv2.imread("misc/frame_550.jpg")
    #visualize the original boxes:
    for index,record in enumerate(frame_result.data):
        point=record[2]
        p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
        #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
        cv2.rectangle(image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=(0,0,255),thickness=3)
    #visualize the matched boxes
    for index,point in enumerate(best_boxes):
        p1_x,p1_y,p2_x,p2_y=int(point[0]),int(point[1]),int(point[2]),int(point[3])
        #print(f"P1： ({p1_x},{p1_y}) P2: ({p2_x},{p2_y})")
        cv2.rectangle(image,pt1=(p1_x,p1_y),pt2=(p2_x,p2_y),color=(0,255,0),thickness=3)
        cv2.putText(image, f"{index}",(p1_x+5, p1_y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imwrite("result_550_visualized.jpg",image)
    print(f"Run time: {end-start}s")
    #return score for parem group
    #return result and best parems
    



