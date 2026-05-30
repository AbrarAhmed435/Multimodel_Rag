import chromadb 


class ImageStore:

    def __init__(self):

        self.client=chromadb.PersistentClient(path="./chroma_db")
        
        self.collection=(self.client.get_or_create_collection(name="image_collection"))


    def add_images(self,ids,embeddings,metadatas):
        self.collection.add(ids=ids,embeddings=embeddings,metadatas=metadatas)
    
    def search(self,query_embedding,k=5):
        return self.collection.query(query_embeddings=[query_embedding],n_results=k)