from embeddings.image_embedder import ImageEmbedder
from vectorstore.image_store import ImageStore


class ImageRetriever:

    def __init__(self):
        self.embedder=ImageEmbedder()
        self.store=ImageStore()

    
    def search(self,query,k=3):
        query_embedding=(self.embedder.embed_text(query))

        results=self.store.search(query_embedding,k)

        print(results)

        output=[]
        metadatas=(results["metadatas"][0])

        distances=(results["distances"][0])

        for meta, dist in zip(metadatas,distances):
            output.append({
                "image_path":meta["image_path"],
                "caption":meta["caption"],
                "page":meta["page"],
                "distance":dist
            })
        return output