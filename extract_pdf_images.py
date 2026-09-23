"""
Convert PDF pages to images for visual inspection and OCR
"""

import fitz  # PyMuPDF
from PIL import Image
import io
import os

def pdf_to_images(pdf_path, output_folder):
    """Convert PDF pages to PNG images"""
    print(f"\nConverting: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page at high resolution
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        output_path = f"{output_folder}/{base_name}_page{page_num + 1}.png"
        pix.save(output_path)
        image_paths.append(output_path)
        print(f"  Saved: {output_path}")
    
    doc.close()
    return image_paths

print("="*80)
print("CONVERTING PDFs TO IMAGES")
print("="*80)

# Convert first PDF
images1 = pdf_to_images("Data_scanned.pdf", "pdf_images")

# Convert second PDF  
images2 = pdf_to_images("Data_scanned_2.pdf", "pdf_images")

print("\n" + "="*80)
print(f"Total images created: {len(images1) + len(images2)}")
print("="*80)
