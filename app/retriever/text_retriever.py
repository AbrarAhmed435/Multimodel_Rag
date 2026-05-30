from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore

class TextRetriever:

    def __init__(self):
        self.embedder=Embedder()
        self.store=ChromaStore()

    def search(self,query:str,k:int=5):
        query_embedding=self.embedder.embed_text(query)
        results=self.store.search(query_embedding,n_results=k)

        documents=results['documents'][0]
        metadatas=results['metadatas'][0]
        distances=results['distances'][0]

        output=[]

        for doc, meta, dist in zip(documents,metadatas,distances):
            output.append({
                "content":doc,
                "metadata":meta,
                "distance":dist
            })
        return output