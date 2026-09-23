"""
Extract data from scanned PDFs
"""

import fitz  # PyMuPDF
import pandas as pd
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    print(f"\nProcessing: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        all_text.append(f"=== Page {page_num + 1} ===\n{text}")
    
    doc.close()
    return "\n\n".join(all_text)

# Extract from first PDF
print("="*80)
print("EXTRACTING DATA FROM SCANNED PDFs")
print("="*80)

text1 = extract_text_from_pdf("Data_scanned.pdf")
print(f"\nData_scanned.pdf - Extracted {len(text1)} characters")

# Save raw text for inspection
with open("extracted_text_scanned1.txt", "w", encoding="utf-8") as f:
    f.write(text1)
print("Saved: extracted_text_scanned1.txt")

# Extract from second PDF
text2 = extract_text_from_pdf("Data_scanned_2.pdf")
print(f"\nData_scanned_2.pdf - Extracted {len(text2)} characters")

# Save raw text for inspection
with open("extracted_text_scanned2.txt", "w", encoding="utf-8") as f:
    f.write(text2)
print("Saved: extracted_text_scanned2.txt")

print("\n" + "="*80)
print("Text extraction complete. Check the .txt files for content.")
print("="*80)
