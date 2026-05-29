from parser.pdf_parser import PDFParser
from parser.document_builder import DocumentBuilder
from embeddings.embedder import Embedder
from vectorstore.chroma_store import ChromaStore
from retriever.retriever import Retriever
from rag.context_builder import ContextBuilder
from rag.prompt_builder import PromptBuilder
from embeddings.image_embedder import ImageEmbedder





def prepare_for_chroms(text_elements,figure_elements,embedder):
    ids=[]
    documents=[]
    embeddings=[]
    metadatas=[]


    #Text Elements
    for idx,element in enumerate(text_elements):

        ids.append(f"text_{idx}")

        documents.append(element.content)

        embeddings.append(embedder.embed_text(element.content))

        metadatas.append({
            "type":"text",
            "page":element.page
        })

    # Figure Elements

    for idx,element in enumerate(figure_elements):
        
        ids.append(f"figure_{idx}")

        documents.append(element.content)

        embeddings.append(embedder.embed_text(element.content))

        metadatas.append({
            "type":"figure",
            "page":element.page,
            "image_path":element.image_path
        })

    return (ids,documents,embeddings,metadatas)

    








def main():

    pdf_path = "app/data/attention.pdf"

    # -----------------------------
    # PDF PARSING
    # -----------------------------
    parser = PDFParser(pdf_path)

    print("\nExtracting text blocks...")
    text_data = parser.extract_text_blocks()

    print("Extracting images...")
    image_data = parser.extract_images()

    print("Matching figure captions...")
    matched_captions = parser.match_captions(
        text_data,
        image_data
    )

    # -----------------------------
    # BUILD DOCUMENT ELEMENTS
    # -----------------------------
    builder = DocumentBuilder()

    print("Building text elements...")
    text_elements = builder.build_text_elements(
        text_data
    )

    print("Building figure elements...")
    figure_elements = builder.build_figure_elements(
        matched_captions
    )

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n==============================")
    print("TEXT ELEMENT SAMPLE")
    print("==============================\n")

    print(text_elements[0])

    print("\n==============================")
    print("FIGURE ELEMENT SAMPLE")
    print("==============================\n")

    if figure_elements:
        print(figure_elements[0])
    else:
        print("No figure elements found.")

    print("\n==============================")
    print("COUNTS")
    print("==============================\n")

    print(f"Text Elements  : {len(text_elements)}")
    print(f"Figure Elements: {len(figure_elements)}")

    embedder = Embedder()

    sample_embedding = embedder.embed_text(
        text_elements[0].content
    )

    # print("\nEmbedding Length:")
    # print(len(sample_embedding))

    # sample1 = text_elements[0].content
    # sample2 = text_elements[1].content

    # emb1 = embedder.embed_text(sample1)
    # emb2 = embedder.embed_text(sample2)

    # print(sample1[:100])
    # print(sample2[:100])

    # print(len(emb1))
    # print(emb1[:10])

    store=ChromaStore()

    ids,docs,embeddings,metadatas=prepare_for_chroms(text_elements,figure_elements,embedder)


    store.add_document(ids,docs,embeddings,metadatas)

    print("Documents inserted into Chroma")

    retriever= Retriever()

    results = retriever.search(
        "What is multi-head attention?"
    )

    # for r in results:
    #     print("\n")
    #     print(r)
    builder=ContextBuilder()
    context=builder.build_context(results)

    # print(context)
    prompt_builder = PromptBuilder()

    prompt = prompt_builder.build_prompt(
        question="What is multi-head attention?",
        context=context
    )
    # print(prompt)
    image_embedder=ImageEmbedder()

    sample_vec=image_embedder.embed_image(figure_elements[0].image_path)

    print(f"Image embedding dimensions:, {len(sample_vec)}")

    print("Embedding\n")
    print(sample_vec[:10])




if __name__ == "__main__":
    main()