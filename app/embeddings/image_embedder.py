from PIL import Image 
from transformers import AutoProcessor, SiglipModel 
import torch 



class ImageEmbedder:

    def __init__(self):
        self.device=("cuda" if torch.cuda.is_available() else "cpu")

        self.processor=AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model=SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)

        self.model.eval()
    
    @torch.no_grad()
    def embed_image(self,image_path):
        image=Image.open(image_path).convert("RGB")
        inputs=self.processor(images=image,return_tensors="pt")

        inputs={
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        image_features=self.model.get_image_features(**inputs)

        # if hasattr(image_features, "pooler_output"):
        #     image_features=image_features.pooler_output

        if not isinstance(image_features,torch.Tensor):
            image_features=image_features.pooler_output



        # print(type(image_features))
        # print(image_features)

        image_features=torch.nn.functional.normalize(image_features,p=2,dim=1)

        return (image_features.cpu().numpy()[0].tolist())
