import fitz
from pathlib import Path 

class PDFParser:
    def __init__(self,pdf_path):
        self.pdf_path=pdf_path 
        self.doc=fitz.open(pdf_path)
    
    def extract_text_blocks(self):
        pages_data=[]

        for page_num,page in enumerate(self.doc):
            text=page.get_text("blocks")

            blocks=[]

            for block in text:
                x0,y0,x1,y1,content,*_=block
                blocks.append({
                    "text":content.strip(),
                    "bbox":[x0,y0,x1,y1]
                })
            pages_data.append({
                "page":page_num+1,
                "blocks":blocks
            })
        return pages_data
    
    def extract_images(self,output_dir="app/output/images"):
        import os 
        os.makedirs(output_dir,exist_ok=True)
        extracted_images=[]
        pdf_name = Path(self.pdf_path).stem
        seen_xrefs = set()

        for page_num,page in enumerate(self.doc):
            image_list=page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref=img[0]

                print(f"Page={page_num+1}\n xref={xref}")

                if xref in seen_xrefs:
                    continue

                seen_xrefs.add(xref)
                
                base_image=self.doc.extract_image(xref)
                image_bytes=base_image["image"]
                image_ext=base_image["ext"]

                image_filename=(f"{pdf_name}_page_{page_num+1}_img_{img_index+1}.{image_ext}")

                image_path=os.path.join(output_dir,image_filename)

                with open(image_path,"wb") as f:
                    f.write(image_bytes)
                
                rects=page.get_image_rects(xref)

                bbox=[]

                if rects:
                    rect=rects[0]

                    bbox={
                        "x0":rect.x0,
                        "y0":rect.y0,
                        "x1":rect.x1,
                        "y1":rect.y1
                    }
                else:
                    bbox=None 

                extracted_images.append({
                    "page":page_num+1,
                    "image_index":img_index+1,
                    "image_path":image_path,
                    "bbox":bbox
                })
        return extracted_images
    
    def match_captions(self,text_blocks,images):
        matched_results=[]

        for image in images:
            page_num=image["page"]

            image_box=image["bbox"]

            if not image_box:
                continue
            
            # image_box=image_box[0]

            image_bottom=image_box["y1"]

            page_data=next((p for p in text_blocks if p["page"]==page_num),None)

            if not page_data:
                continue
            best_caption=None
            min_distance=float("inf")

            for block in page_data["blocks"]:
                text=block["text"]

                if "Figure" not in text and "Fig." not in text:
                    continue
                block_y0=block["bbox"][1]

                if block_y0> image_bottom:
                    distance=block_y0-image_bottom

                    if distance<min_distance:
                        min_distance=distance
                        best_caption=text 
            matched_results.append({
                "page":page_num,
                "image_path":image["image_path"],
                "caption":best_caption
            })
        return matched_results