from models.document_elements import TextElement, FigureElement

class DocumentBuilder:


    def build_text_elements(self,text_blocks,document_name):

        elements=[]

        for page_data in text_blocks:
            page_num=page_data["page"]
            for block in page_data["blocks"]:
                text=block["text"].strip()

                if not text:
                    continue

                element=TextElement(
                    type="text",
                    document_name=document_name,
                    page=page_num,
                    content=text,
                    bbox=block['bbox']
                )

                elements.append(element)

        return elements


    def build_figure_elements(self,matched_captions,document_name):

        figures=[]

        for item in matched_captions:
            element=FigureElement(
                type="figure",
                document_name=document_name,
                page=item["page"],
                content=item["caption"] or "",
                caption=item["caption"],
                image_path=item["image_path"]
            )

            figures.append(element)

        return figures
