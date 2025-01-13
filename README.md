<a href="https://github.com/imanoop7/Ollama-OCR"><img src="https://img.shields.io/github/stars/imanoop7/Ollama-OCR.svg?style=social&label=Star" alt="Stargazers"></a>
<a href="https://github.com/imanoop7/Ollama-OCR/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/imanoop7/Ollama-OCR.svg" alt="Commit Activity"></a>
<a href="https://github.com/imanoop7/Ollama-OCR"><img src="https://img.shields.io/github/last-commit/imanoop7/Ollama-OCR.svg" alt="Last Commit"></a>
<a href="https://github.com/imanoop7/Ollama-OCR/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/imanoop7/Ollama-OCR.svg" alt="Contributors"></a>
# Ollama OCR 🔍

A powerful OCR (Optical Character Recognition) package that uses state-of-the-art vision language models through Ollama to extract text from images. Available both as a Python package and a Streamlit web application.

## 🌟 Features

- **Multiple Vision Models Support**
  - LLaVA 7B: Efficient vision-language model for real-time processing (LLaVa model can generate wrong output sometimes)
  - Llama 3.2 Vision: Advanced model with high accuracy for complex documents

- **Multiple Output Formats**
  - Markdown: Preserves text formatting with headers and lists
  - Plain Text: Clean, simple text extraction
  - JSON: Structured data format
  - Structured: Tables and organized data
  - Key-Value Pairs: Extracts labeled information

- **Batch Processing**
  - Process multiple images in parallel
  - Progress tracking for each image
  - Image preprocessing (resize, normalize, etc.)


## 📦 Package Installation

```bash
pip install ollama-ocr
```

## 🚀 Quick Start
### Prerequisites
1. Install Ollama
2. Pull the required model:

```bash
ollama pull llama3.2-vision:11b
```
## Using the Package

### Single Image Processing

```python
from ollama_ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(model_name='llama3.2-vision:11b')  # You can use any vision model available on Ollama

# Process an image
result = ocr.process_image(
    image_path="path/to/your/image.png",
    format_type="markdown"  # Options: markdown, text, json, structured, key_value
)
print(result)
```
### Batch Processing (New! 🆕)

```python
from ollama_ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(model_name='llama3.2-vision:11b', max_workers=4)  # max workers for parallel processing

# Process multiple images
# Process multiple images with progress tracking
batch_results = ocr.process_batch(
    input_path="path/to/images/folder",  # Directory or list of image paths
    format_type="markdown",
    recursive=True,  # Search subdirectories
    preprocess=True  # Enable image preprocessing
)
# Access results
for file_path, text in batch_results['results'].items():
    print(f"\nFile: {file_path}")
    print(f"Extracted Text: {text}")

# View statistics
print("\nProcessing Statistics:")
print(f"Total images: {batch_results['statistics']['total']}")
print(f"Successfully processed: {batch_results['statistics']['successful']}")
print(f"Failed: {batch_results['statistics']['failed']}")
```



## 📋 Output Format Details

1. **Markdown Format**: The output is a markdown string containing the extracted text from the image.
2. **Text Format**: The output is a plain text string containing the extracted text from the image.
3. **JSON Format**: The output is a JSON object containing the extracted text from the image.
4. **Structured Format**: The output is a structured object containing the extracted text from the image.
5. **Key-Value Format**: The output is a dictionary containing the extracted text from the image.  

-----
## 🌐 Streamlit Web Application(supports batch processing)
- **User-Friendly Interface**
  - Drag-and-drop image upload
  - Real-time processing
  - Download extracted text
  - Image preview with details
  - Responsive design

1. Clone the repository:
```bash
git clone https://github.com/imanoop7/Ollama-OCR.git
cd Ollama-OCR
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Go to the directory where app.py is located:
```bash
cd src/ollama_ocr      
```
3. Run the Streamlit app:
```bash
streamlit run app.py
```


## 🙏 Acknowledgments
Built with Ollama
Powered by LLaMA Vision Models


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=imanoop7/Ollama-OCR&type=Date)](https://star-history.com/#imanoop7/Ollama-OCR&Date)

