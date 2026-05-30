from embeddings.image_embedder import ImageEmbedder
from vectorstore.image_store import ImageStore



class ImagePipeline:

    def __init__(self):

        self.embedder=ImageEmbedder()
        self.store=ImageStore()

    
    def ingest_images(self,figure_elements):
        image_ids=[]
        image_embeddings=[]
        image_metadatas=[]

        for idx, figure in enumerate(figure_elements):
            image_vec=self.embedder.embed_image(figure.image_path)

            image_ids.append(f"{figure.document_name}_img_{idx}")

            image_embeddings.append(image_vec)

            image_metadatas.append({
                "document_name":figure.document_name,
                "page":figure.page,
                "image_path":figure.image_path,
                "caption":figure.caption
            })

        self.store.add_images(image_ids,image_embeddings,image_metadatas)

        return len(image_ids)

