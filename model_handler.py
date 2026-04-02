import os

import torch
from PIL import Image
from typing import List, Dict
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Local default: smaller 2B checkpoint (override with env COSMOS_MODEL)
DEFAULT_COSMOS_MODEL = "nvidia/Cosmos-Reason2-2B"


class CosmosModelHandler:
    """Vision-language captions per frame using NVIDIA Cosmos Reason2 (default: 2B)."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize the Cosmos model.

        Args:
            model_name: HuggingFace model id or local path. Defaults to COSMOS_MODEL env or Cosmos-Reason2-2B.
        """
        _env_model = (os.getenv("COSMOS_MODEL") or "").strip()
        self.model_name = (model_name or _env_model or DEFAULT_COSMOS_MODEL).strip() or DEFAULT_COSMOS_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}; model: {self.model_name}")
        
        # Load model and processor
        # Note: Adjust these based on actual Cosmos model requirements
        try:
            print("Loading Cosmos model...")
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="eager"
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have access to the model and correct model name.")
            raise
    
    # def analyze_single_frame(
    #     self,
    #     image: Image.Image,
    #     prompt: str = "Describe what is happening in this image in detail."
    # ) -> str:
    #     """
    #     Analyze a single frame with the Cosmos model
        
    #     Args:
    #         image: PIL Image to analyze
    #         prompt: Text prompt for the model
            
    #     Returns:
    #         Model's description of the frame
    #     """
    #     try:
    #         # Prepare inputs
    #         # Note: Adjust based on actual Cosmos model input format
    #         inputs = self.processor(
    #             text=prompt,
    #             images=image,
    #             return_tensors="pt"
    #         ).to(self.device)
            
    #         # Generate response
    #         with torch.no_grad():
    #             output = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=150,
    #                 do_sample=False,
    #                 temperature=0.7,
    #                 top_p=0.9
    #             )
            
    #         # Decode output
    #         response = self.processor.decode(output[0], skip_special_tokens=True)
            
    #         # Extract only the generated text (remove prompt)
    #         if prompt in response:
    #             response = response.split(prompt)[-1].strip()
            
    #         return response
            
    #     except Exception as e:
    #         print(f"Error analyzing frame: {e}")
    #         return f"Error: Could not analyze frame - {str(e)}"

    def analyze_single_frame(self, image: Image.Image, prompt: str = "Describe what is happening in this image in detail.") -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False
                )

            # Decode only the newly generated tokens
            generated_ids = output[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response

        except Exception as e:
            print(f"Error analyzing frame: {e}")
            return f"Error: Could not analyze frame - {str(e)}"

    
    def analyze_frames(
        self,
        frames: List[Image.Image],
        batch_size: int = 1
    ) -> List[Dict[str, str]]:
        """
        Analyze multiple frames
        
        Args:
            frames: List of PIL Images
            batch_size: Number of frames to process at once (for efficiency)
            
        Returns:
            List of dictionaries with frame descriptions
        """
        descriptions = []
        
        for idx, frame in enumerate(frames):
            print(f"Analyzing frame {idx + 1}/{len(frames)}...")
            
            # Create a contextual prompt
            prompt = (
                f"You are analyzing frame {idx + 1} of a video. "
                "Describe what is happening in this image, including: "
                "1) The main subjects or objects, "
                "2) The action or activity taking place, "
                "3) The setting or environment, "
                "4) Any notable details. "
                "Be concise but informative."
            )
            
            description = self.analyze_single_frame(frame, prompt)
            
            descriptions.append({
                "frame_index": idx,
                "description": description
            })
        
        return descriptions
    
    def analyze_with_context(
        self,
        frames: List[Image.Image],
        previous_context: str = ""
    ) -> List[Dict[str, str]]:
        """
        Analyze frames with context from previous frames
        
        Args:
            frames: List of PIL Images
            previous_context: Summary of what happened in previous frames
            
        Returns:
            List of dictionaries with frame descriptions
        """
        descriptions = []
        context = previous_context
        
        for idx, frame in enumerate(frames):
            print(f"Analyzing frame {idx + 1}/{len(frames)} with context...")
            
            # Build prompt with context
            if context:
                prompt = (
                    f"Previously in this video: {context}\n\n"
                    f"Now, describe what is happening in this new frame. "
                    "Focus on what has changed or what is new."
                )
            else:
                prompt = (
                    "This is the first frame of a video. "
                    "Describe what you see in detail."
                )
            
            description = self.analyze_single_frame(frame, prompt)
            
            descriptions.append({
                "frame_index": idx,
                "description": description
            })
            
            # Update context for next frame
            context = description
        
        return descriptions
    
    def cleanup(self):
        """Free up GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
