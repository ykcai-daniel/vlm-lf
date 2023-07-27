import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
#https://huggingface.co/docs/transformers/main/model_doc/owlvit

class Clip:
    def __init__(self) -> None:
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #note: image-to-text logits transpose is text-to-image logits
    def get_logits(self,image,text_queries):
        inputs=self.processor(text=text_queries, images=image, return_tensors="pt",padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            #we use absolute value as result?
            #probs = logits_per_image.softmax(dim=1) 
        return logits_per_image

    #image feat and text feat still needs to be projected
    def get_image_feat(self,image):
        with torch.no_grad():
            inputs=self.processor(images=image,return_tensors='pt')
            return self.model.get_image_features(**inputs)
        
    #image feat and text feat still needs to be projected 
    def get_text_feat(self,text:str):
        with torch.no_grad():
            inputs=self.processor.tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors='pt')
            return self.model.get_text_features(**inputs)

if __name__=='__main__':
    #max precision given different threshold criteria
    clip=Clip()
    img=cv2.imread(f'./data/astronaut.jpg')
    texts=['woman','astronaut','airplane','computer','handbag']
    logits=clip.get_logits(img,texts)
    print(logits)