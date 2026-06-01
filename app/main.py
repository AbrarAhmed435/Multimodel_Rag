from pathlib import Path

from parser.pdf_parser import PDFParser
from parser.document_builder import DocumentBuilder

from pipeline.text_pipeline import TextPipeline
from pipeline.image_pipeline import ImagePipeline

from retriever.hybrid_retriever import HybridRetriever

from rag.context_builder import ContextBuilder
from rag.prompt_builder import PromptBuilder


from llm.qwen_vl import QwenVL




def main():

    # =====================================
    # CONFIG
    # =====================================

    pdf_path = "app/data/attention.pdf"
    document_name = Path(pdf_path).stem

    # =====================================
    # PARSE PDF
    # =====================================

    parser = PDFParser(pdf_path)

    print("Extracting text blocks...")
    text_data = parser.extract_text_blocks()

    print("Extracting images...")
    image_data = parser.extract_images()

    print("Matching captions...")
    matched_captions = parser.match_captions(
        text_data,
        image_data
    )

    # =====================================
    # BUILD ELEMENTS
    # =====================================

    builder = DocumentBuilder()

    text_elements = builder.build_text_elements(
        text_data,
        document_name
    )

    figure_elements = builder.build_figure_elements(
        matched_captions,
        document_name
    )

    print(
        f"Text Elements: {len(text_elements)}"
    )

    print(
        f"Figure Elements: {len(figure_elements)}"
    )

    # =====================================
    # INGEST TEXT
    # =====================================

    text_pipeline = TextPipeline()

    text_count = text_pipeline.ingest(
        text_elements,
        figure_elements
    )

    print(
        f"Stored {text_count} text records"
    )

    # =====================================
    # INGEST IMAGES
    # =====================================

    image_pipeline = ImagePipeline()

    image_count = image_pipeline.ingest_images(
        figure_elements
    )

    print(
        f"Stored {image_count} images"
    )

    # =====================================
    # RETRIEVAL
    # =====================================

    query = "What is multi-head attention?"

    hybrid_retriever = HybridRetriever()

    results = hybrid_retriever.search(
        query
    )

    # =====================================
    # BUILD CONTEXT
    # =====================================

    context_builder = ContextBuilder()

    context_text, image_paths = context_builder.build_context(results)

    qwen=QwenVL()

    answer=qwen.generate_answer(question=query,context_text=context_text,image_paths=image_paths)

    # =====================================
    # BUILD PROMPT
    # =====================================

    # prompt_builder = PromptBuilder()

    # prompt = prompt_builder.build_prompt(
    #     question=query,
    #     context=context_text
    # )

    print(answer)

    # =====================================
    # OUTPUT
    # =====================================
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    # print(context_text)

    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    # print("\n" + "=" * 60)
    # print("PROMPT")
    # print("=" * 60)

    # print(prompt)

    # print("\n" + "=" * 60)
    # print("IMAGES")
    # print("=" * 60)

    # for path in image_paths:
    #     print(path)


if __name__ == "__main__":
    main()