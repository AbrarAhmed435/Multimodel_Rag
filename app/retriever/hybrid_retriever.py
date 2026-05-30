from retriever.text_retriever import TextRetriever 
from retriever.image_retriever import ImageRetriever



class HybridRetriever:

    def __init__(self):
        self.text_retriever=TextRetriever() 
        self.image_retriever=ImageRetriever()

    
    def search(self,query,text_k=5,image_k=3):
        text_results=self.text_retriever.search(query)

        image_results=(self.image_retriever.search(query,image_k))

        return {
            "text_results":text_results,
            "image_results":image_results
        }