from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore


class TextPipeline:
    def __init__(self):
        self.embedder=Embedder()
        self.store=ChromaStore()

    def ingest(self,text_elements,figure_elements):
        ids=[]
        documents=[]
        embeddings=[]
        metadatas=[]


        for idx,element in enumerate(text_elements):
            ids.append(f"{element.document_name}_text_{idx}")

            documents.append(element.content)
            embeddings.append(self.embedder.embed_text(element.content))

            metadatas.append({
                "type":"text",
                "document_name":element.document_name,
                "page":element.page
            })
        

        #FIGURE CAPTIONS
        for idx,element in enumerate(figure_elements):
            ids.append(f"{element.document_name}_figure_{idx}")
            documents.append(element.content)

            embeddings.append(self.embedder.embed_text(element.content))

            metadatas.append({
                "type":"figure",
                "document_name":element.document_name,
                "page":element.page,
                "image_path":element.image_path
            })

        self.store.add_document(ids,documents,embeddings,metadatas)

        return len(ids)