import json
from typing import Dict, Any, List, Union
import os
import base64
import requests
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import cv2
from pdf2image import convert_from_path

class OCRProcessor:
    def __init__(self, model_name: str = "llama3.2-vision:latest", 
                 base_url: str = "http://localhost:11434/api/generate",
                 max_workers: int = 1):
        
        self.model_name = model_name
        self.base_url = base_url
        self.max_workers = max_workers

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image before OCR:
        - Convert PDF to image if needed
        - Auto-rotate
        - Enhance contrast
        - Reduce noise
        """
        # Handle PDF files
        if image_path.lower().endswith('.pdf'):
            pages = convert_from_path(image_path)
            if not pages:
                raise ValueError("Could not convert PDF to image")
            # Save first page as temporary image
            temp_path = f"{image_path}_temp.jpg"
            pages[0].save(temp_path, 'JPEG')
            image_path = temp_path

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Auto-rotate if needed
        # TODO: Implement rotation detection and correction

        # Save preprocessed image
        preprocessed_path = f"{image_path}_preprocessed.jpg"
        cv2.imwrite(preprocessed_path, denoised)

        return preprocessed_path

    def process_image(self, image_path: str, format_type: str = "markdown", preprocess: bool = True) -> str:
        """
        Process an image and extract text in the specified format
        
        Args:
            image_path: Path to the image file
            format_type: One of ["markdown", "text", "json", "structured", "key_value"]
            preprocess: Whether to apply image preprocessing
        """
        try:
            if preprocess:
                image_path = self._preprocess_image(image_path)
            
            image_base64 = self._encode_image(image_path)
            
            # Clean up temporary files
            if image_path.endswith(('_preprocessed.jpg', '_temp.jpg')):
                os.remove(image_path)

            # Generic prompt templates for different formats
            # prompts = {
            #     "markdown": """Please look at this image and extract all the text content. Format the output in markdown:
            #     - Use headers (# ## ###) for titles and sections
            #     - Use bullet points (-) for lists
            #     - Use proper markdown formatting for emphasis and structure
            #     - Preserve the original text hierarchy and formatting as much as possible""",

            #     "text": """Please look at this image and extract all the text content. 
            #     Provide the output as plain text, maintaining the original layout and line breaks where appropriate.
            #     Include all visible text from the image.""",

            #     "json": """Please look at this image and extract all the text content. Structure the output as JSON with these guidelines:
            #     - Identify different sections or components
            #     - Use appropriate keys for different text elements
            #     - Maintain the hierarchical structure of the content
            #     - Include all visible text from the image""",

            #     "structured": """Please look at this image and extract all the text content, focusing on structural elements:
            #     - Identify and format any tables
            #     - Extract lists and maintain their structure
            #     - Preserve any hierarchical relationships
            #     - Format sections and subsections clearly""",

            #     "key_value": """Please look at this image and extract text that appears in key-value pairs:
            #     - Look for labels and their associated values
            #     - Extract form fields and their contents
            #     - Identify any paired information
            #     - Present each pair on a new line as 'key: value'"""
            # }
            prompts = {
                    "markdown": """Please analyze this image and extract all information related to technical graph drawings. Format the output in markdown:
                    - Use headers (# ## ###) for key sections (e.g., Axes, Dimensions, Calculations)
                    - Include details like axis labels, length, breadth, height, width, and area
                    - Use bullet points (-) for listing measurements or attributes
                    - Maintain the hierarchy and formatting of the extracted information in markdown format""",

                    "text": """Please analyze this image and extract all information related to technical graph drawings. 
                    Provide the output as plain text, including:
                    - Details about axis labels and scales
                    - Dimensions such as length, breadth, height, and width
                    - Calculations like area or volume if mentioned
                    - Maintain the original layout and line breaks for clarity.""",

                    "json": """Please analyze this image and extract all information related to technical graph drawings. Structure the output as JSON with these guidelines:
                    - Use keys like 'Axes', 'Dimensions', 'Calculations', 'Scales'
                    - Include axis labels, scales, and orientation under 'Axes'
                    - Capture length, breadth, height, width, and area under 'Dimensions'
                    - Maintain a hierarchical structure to represent the technical graph details
                    - Include all relevant information in the JSON output""",

                    "structured": """Please analyze this image and extract all information related to technical graph drawings, focusing on structural elements:
                    - Identify and format any tables or diagrams that describe graph dimensions
                    - Extract axis details, labels, and scales
                    - Include structural details like length, breadth, height, width, and area
                    - Clearly format sections for easy readability and analysis""",

                    "key_value": """Please analyze this image and extract technical graph drawing details in key-value pairs:
                    - Extract axis information as 'Axis: Value' (e.g., 'X-Axis: 10 cm')
                    - List dimensions such as 'Length: Value', 'Breadth: Value', etc.
                    - Include additional properties like area as 'Area: Value'
                    - Present each pair on a new line for clarity"""
                }


            # Get the appropriate prompt
            prompt = prompts.get(format_type, prompts["text"])

            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "images": [image_base64]
            }

            # Make the API call to Ollama
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            result = response.json().get("response", "")
            
            # Clean up the result if needed
            if format_type == "json":
                try:
                    # Try to parse and re-format JSON if it's valid
                    json_data = json.loads(result)
                    return json.dumps(json_data, indent=2)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw result
                    return result
            
            return result
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def process_batch(
        self,
        input_path: Union[str, List[str]],
        format_type: str = "markdown",
        recursive: bool = False,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple images in batch
        
        Args:
            input_path: Path to directory or list of image paths
            format_type: Output format type
            recursive: Whether to search directories recursively
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Dictionary with results and statistics
        """
        # Collect all image paths
        image_paths = []
        if isinstance(input_path, str):
            base_path = Path(input_path)
            if base_path.is_dir():
                pattern = '**/*' if recursive else '*'
                for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.tiff']:
                    image_paths.extend(base_path.glob(f'{pattern}{ext}'))
            else:
                image_paths = [base_path]
        else:
            image_paths = [Path(p) for p in input_path]

        results = {}
        errors = {}
        
        # Process images in parallel with progress bar
        with tqdm(total=len(image_paths), desc="Processing images") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_image, str(path), format_type, preprocess): path
                    for path in image_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        results[str(path)] = future.result()
                    except Exception as e:
                        errors[str(path)] = str(e)
                    pbar.update(1)

        return {
            "results": results,
            "errors": errors,
            "statistics": {
                "total": len(image_paths),
                "successful": len(results),
                "failed": len(errors)
            }
        }