import pymupdf4llm
import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Configure the Gemini Pro API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the GenerativeModel for Gemini Pro
model = genai.GenerativeModel(model_name='gemini-pro')

# Function to extract text from PDF using pymupdf4llm
def extract_text_from_pdf(pdf_path):
    try:
        # Extract text from PDF using pymupdf4llm
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Take the first 100 words from md_text
        first_100_words = ' '.join(md_text.split()[:100])
        
        # Write extracted text to a text file
        output_file_path = 'Pymupdf4llm_output1.txt'
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(first_100_words)
        
        print(f"Extracted text (first 100 words) saved to {output_file_path}")
        
        return first_100_words
    except ValueError as e:
        print(f"Error processing PDF: {e}")
        return ""

# Function to generate question-answer pairs using Gemini Pro
def generate_qa_pairs(text):
    prompt = f"Generate question-answer pairs from the following text:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text.strip()

# Example usage:
pdf_file_path = r"C:/Users/prasannakt/Desktop/Deploy_Rag/wipro_policy.pdf"
extracted_text = extract_text_from_pdf(pdf_file_path)

# Split text into pages (assuming each page is separated by "\n\n" in the extracted text)
pages = extracted_text.split("\n\n")

# Generate question-answer pairs for each page
qa_pairs = ""
for page_num, page_text in enumerate(pages):
    qa_pairs += f"Page {page_num + 1}:\n"
    qa_pairs += generate_qa_pairs(page_text) + "\n\n"

# Save QA pairs to a text file with UTF-8 encoding
qa_file_path = 'qa_pairs1.txt'
with open(qa_file_path, 'w', encoding='utf-8') as qa_file:
    qa_file.write(qa_pairs)

print("Question-Answer pairs generated and saved successfully!")
