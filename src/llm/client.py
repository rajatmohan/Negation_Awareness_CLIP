"""
Local Qwen LLM client for batch inference.
Handles tokenization, batch processing, and JSON extraction.
"""

import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwenClient:
    """
    Local inference wrapper for Qwen2.5 model.
    Supports batch processing with left-padding for decoder-only models.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        self.device = device
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # CRITICAL: Decoder-only models must pad on the left
        self.tokenizer.padding_side = "left"
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device
        )

    def generate_batch(self, system_prompt, queries):
        """
        Process multiple queries in a single GPU forward pass.
        
        Args:
            system_prompt: System instruction for LLM
            queries: List of query strings
        
        Returns:
            List of parsed JSON responses
        """
        formatted_texts = []
        for q in queries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Query: "{q}"'}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_texts.append(text)

        inputs = self.tokenizer(
            formatted_texts, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        responses = []
        input_len = inputs.input_ids.shape[1]
        batch_output_ids = generated_ids[:, input_len:]
        decoded_outputs = self.tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)

        for i, response in enumerate(decoded_outputs):
            try:
                # Extract JSON from response
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    clean_json = match.group(0)
                    responses.append(json.loads(clean_json))
                else:
                    raise ValueError("No JSON block found")
                    
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Extraction Error for: {queries[i][:30]}... using fallback.")
                # Fallback response
                responses.append({
                    "positives": [queries[i]], 
                    "negatives": []
                })
        
        return responses

    def __call__(self, system_prompt, user_prompt):
        """Backward compatibility for single query"""
        return self.generate_batch(system_prompt, [user_prompt])[0]
