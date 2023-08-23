from groundingdino.util.inference import load_model, load_image, predict, annotate

BOX_TRESHOLD = 0.4
TEXT_TRESHOLD = 0.25


#TODO: extract a parent class
class DinoProcessor:
    def __init__(self,box_thresh=BOX_TRESHOLD,text_thresh=TEXT_TRESHOLD) -> None:
        config_file="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        checkpoint = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.model = load_model(config_file, checkpoint)
        self.box_thresh=box_thresh
        self.text_thresh=text_thresh

    def process_image(self,image_path:str,text_queries:str,device='cpu'):
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_queries,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        print(boxes,logits,phrases)
        results={
            'scores':[],
            'labels':[],
            'logits':[],
            'boxes':[]
        }
        return results
    
    
    def image_query(self,image,image_query,device='cpu'):
        raise NotImplementedError


if __name__=="__main__":
    dino_processor=DinoProcessor()
    result=dino_processor.process_image("data/frame_550.jpg","a man with two suitcases")