from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
BOX_TRESHOLD = 0.4
TEXT_TRESHOLD = 0.25
import torchvision
import cv2

#TODO: extract a parent class
class DinoProcessor:
    def __init__(self,box_thresh=BOX_TRESHOLD,text_thresh=TEXT_TRESHOLD) -> None:
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.model = load_model(config_file, checkpoint)
        self.box_thresh=box_thresh
        self.text_thresh=text_thresh

    #TODO: support CPU
    def process_image(self,image_path:str,text_queries:str,visualize=False):
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_queries,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        h, w, _ = image_source.shape
        boxes_scaled = boxes * torch.Tensor([w, h, w, h])
        xyxy = torchvision.ops.box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        results={
            'scores':logits.tolist(),
            'labels':torch.zeros_like(logits).tolist(),
            'logits':logits.tolist(),
            'boxes':xyxy.tolist()
        }
        print(results)
        if visualize:
            return results,annotate(image_source,boxes,logits,phrases)
        else:
            return results,None
    
    
    def image_query(self,image,image_query,device='cpu'):
        raise NotImplementedError


if __name__=="__main__":
    dino_processor=DinoProcessor()
    result,annotated_frame=dino_processor.process_image("misc/frame_550.jpg","A man with two suitcases .",visualize=True)
    cv2.imwrite("dino_example.jpg",annotated_frame)