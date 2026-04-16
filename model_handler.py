import torch
import os
from PIL import Image
from typing import List, Dict
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration



class CosmosModelHandler:
    """Handles interaction with Nvidia's Cosmos-reason2-8b model"""
    
    def __init__(self, model_name: str = "nvidia/Cosmos-Reason2-8b"):
        """
        Initialize the Cosmos model
        
        Args:
            model_name: HuggingFace model identifier or local path
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        # Note: Adjust these based on actual Cosmos model requirements
        try:
            print("Loading Cosmos model...")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="eager",
                token=os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have access to the model and correct model name.")
            raise
    

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
