from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
import torch

class QwenVL:
    def __init__(self):
        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="auto",
                device_map="auto"
            )
        )
        self.processor = (
            AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct"
            )
        )

    def generate_answer(
        self,
        question,
        context_text,
        image_paths
    ):
        prompt = f"""
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not available,
say that the information is not available.
Context:
{context_text}
Question:
{question}
"""
        content = []
        for path in image_paths:
            content.append(
                {
                    "type": "image",
                    "image": path
                }
            )
        content.append(
            {
                "type": "text",
                "text": prompt
            }
        )
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = (
            process_vision_info(messages)
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids
            in zip(
                inputs.input_ids,
                generated_ids
            )
        ]
        output_text = (
            self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        )
        return output_text[0]