class ContextBuilder:
    
    def build_context(self,hybrid_results):
        text_results=hybrid_results["text_results"]
        image_results=hybrid_results["image_results"]

        context_parts=[]
        image_paths=[]



        for idx,result in enumerate(text_results,start=1):
            context_parts.append(f"""
            [Chunks {idx}]
            Page: {result["metadata"]["page"]}
            Type: {result["metadata"]["type"]}

            {result["content"]}

            """)


        seen_paths=set() 

        for image in image_results:

            path=image["image_path"]

            if path in seen_paths:
                continue

            seen_paths.add(path)

            image_paths.append(path)

            context_text="\n".join(context_parts)

            return context_text,image_paths[:3]