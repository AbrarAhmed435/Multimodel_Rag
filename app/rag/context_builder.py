class ContextBuilder:
    def build_context(self,retrieved_docs):
        context_parts=[]
        for idx,doc in enumerate(retrieved_docs,start=1):
            page=doc["metadata"].get("page","unknown")

            doc_type=doc["metadata"].get("type","unknown")

            content=doc["content"]

            chunk=f"""
            [Chunk {idx}]
            Page: {page}
            Type: {doc_type}

            {content}
            """

            context_parts.append(chunk)

        return "\n".join(context_parts)