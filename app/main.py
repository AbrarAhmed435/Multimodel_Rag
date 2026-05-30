from parser.pdf_parser import PDFParser
from parser.document_builder import DocumentBuilder 

from pipeline.text_pipeline import TextPipeline
from pipeline.image_pipeline import ImagePipeline

from retriever.text_retriever import TextRetriever 
from rag.context_builder import ContextBuilder
from retriever.image_retriever import ImageRetriever
from rag.prompt_builder import PromptBuilder 
from retriever.hybrid_retriever import HybridRetriever


from pathlib import Path 


def main():

    pdf_path="app/data/attention.pdf"

    document_name=Path(pdf_path).stem


    parser=PDFParser(pdf_path)

    print("Extracting text blocks....")

    text_data=parser.extract_text_blocks()

    print("Extracting images... ")
    image_data=parser.extract_images()

    print("Matching figure captions...")
    matched_captions=parser.match_captions(text_data,image_data)

    

    builder=DocumentBuilder()

    print("Building text elements....")

    text_elements=builder.build_text_elements(text_data,document_name)


    print("Building figure element....")

    figure_elements=builder.build_figure_elements(matched_captions,document_name)

    print("\n################ COUNTS ###############\n")
    print(f"Text Elements: {len(text_elements)}")

    print(f"Figure Elements: {len(figure_elements)}")


    text_pipeline=TextPipeline()

    text_count=text_pipeline.ingest(text_elements,figure_elements)

    print(f"Stored {text_count} text records")

    image_pipeline=ImagePipeline()

    image_count=image_pipeline.ingest_images(figure_elements)

    
    print("IMAGE PIPELINE STORE COLLECTION COUNT \n")
    print(image_pipeline.store.collection.count())


    print(f"Stored {image_count} images")


    # retriever=Retriever()

    # results=retriever.search("What is multihead attention?")

    # context_builder=ContextBuilder()

    # context=context_builder.build_context(results)
    
    # prompt_builder=PromptBuilder()

    # prompt=prompt_builder.build_prompt(question="What is multi-head attention",context=context)

    # print("###################### PROMPT ###########################")

    # print(prompt)
    # query_vec = image_pipeline.embedder.embed_text(
    #     "multi head attention"
    # )

    # # print(
    # #     len(query_vec)
    # # )

    # image_retriever = ImageRetriever()

    # results = image_retriever.search(
    #     "multi head attention"
    # )

    # print("############### RESULTS ###########")

    # for r in results:
    #     print("\n")
    #     print(r)

    hybrid_retriever = HybridRetriever()

    results = hybrid_retriever.search(
        "What is multi-head attention?"
    )

    print("\nTEXT RESULTS")
    print("=" * 50)

    for item in results["text_results"]:
        print(item)

    print("\nIMAGE RESULTS")
    print("=" * 50)

    for item in results["image_results"]:
        print(item)


if __name__=="__main__":
    main()



