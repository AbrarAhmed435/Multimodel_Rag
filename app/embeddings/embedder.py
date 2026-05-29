from sentence_transformers import SentenceTransformer



class Embedder:
    def __init__(self):
        self.model=SentenceTransformer("BAAI/bge-base-en-v1.5")

    def embed_text(self,text):
        embedding=self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()