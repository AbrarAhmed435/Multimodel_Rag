import chromadb

class ChromaStore:
    
    def __init__(self):

        self.client=chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection=self.client.get_or_create_collection(
            name="pdf_elements"
        )

    def add_document(self,ids,documents,embeddings,metadatas):
        self.collection.add(ids=ids,documents=documents,embeddings=embeddings,metadatas=metadatas)
    
    def search(self,query_embedding,n_results=5):
        results=self.collection.query(query_embeddings=[query_embedding],n_results=n_results)

        return results