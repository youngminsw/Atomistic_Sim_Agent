import os
import google.generativeai as genai
from src.config import Config

class GeminiClient:
    def __init__(self):
        # Assuming Config has GEMINI_API_KEY
        # If not, we fall back to os.environ or a placeholder
        api_key = getattr(Config, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
        if not api_key:
             print("[GeminiClient] Warning: GEMINI_API_KEY not found in Config.")
        else:
            genai.configure(api_key=api_key)
            
        # User requested "Gemini 3 pro high", mapping to current avail flagship "gemini-1.5-pro-latest"
        self.model_name = getattr(Config, "GEMINI_MODEL", "gemini-1.5-pro-latest")
        self.model = genai.GenerativeModel(self.model_name)

    def generate_json(self, prompt, images=None):
        """
        Generates JSON response from Gemini, supporting multimodal input.
        :param prompt: Text prompt
        :param images: List of image file paths
        """
        try:
            content = [prompt]
            
            if images:
                import PIL.Image
                for img_path in images:
                    if os.path.exists(img_path):
                        content.append(PIL.Image.open(img_path))
                    else:
                        print(f"   [Gemini-Client] Warning: Image not found at {img_path}")

            # Enforce JSON structure in prompt/config if supported
            generation_config = {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
            
            response = self.model.generate_content(
                content,
                generation_config=generation_config
            )
            
            return response.text # Should be JSON string
            
        except Exception as e:
            print(f"   [Gemini-Client] Error: {e}")
            import traceback
            traceback.print_exc()
            return None
