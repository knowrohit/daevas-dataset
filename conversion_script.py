import re
import nltk
# nltk.download('punkt')
import pandas as pd
import fitz  # PyMuPDF
from typing import List, Tuple

def convert_pdf_to_txt(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_by_sentences(text: str) -> List[str]:
    sentences = re.split(r'[.!?]', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def create_overlapping_chunks(lst: List[str], chunk_size: int, overlap: int) -> List[Tuple]:
    chunks = []
    for i in range(0, len(lst) - chunk_size + 1, overlap):
        chunks.append((i+1, i+chunk_size, ' '.join(lst[i:i+chunk_size])))
    return chunks

def process_textbook(file_path: str, chunk_size: int, overlap: int) -> pd.DataFrame:
    text = convert_pdf_to_txt(file_path)
    print(f"PDF content: {text[:500]}")  # Print the first 500 characters of the PDF content

    sentences = split_by_sentences(text)
    print(f"Number of sentences: {len(sentences)}")  # Print the number of sentences
    print(f"First 5 sentences: {sentences[:5]}")  # Print the first 5 sentences

    # Define a dictionary to map chapter numbers in words to numeric values
    chapter_number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    print(chapter_number_words)

    # Extract chapter number from the first sentence
    chapter_number = "1"  # Default value
    chapter_number_match = re.search(r'Chapter\s*(\w+)', text)
    if chapter_number_match:
        chapter_number_word = chapter_number_match.group(1).strip().lower() 
        chapter_number = chapter_number_words.get(chapter_number_word, "1")
        print(f"Chapter number word: {chapter_number_word}")
        print(f"Chapter number: {chapter_number}")
        print(chapter_number_words.keys())

    chunks = create_overlapping_chunks(sentences, chunk_size, overlap)
    print(f"Number of chunks: {len(chunks)}")  # Print the number of chunks
    print(f"First 5 chunks: {chunks[:5]}")  # Print the first 5 chunks

    rows = []
    for start, end, chunk_text in chunks:
        rows.append({
            'Chapter': int(chapter_number),
            'sentence_range': f'{start}-{end}',
            'Text': chunk_text
        })
    df = pd.DataFrame(rows)
    print(f"DataFrame head: {df.head()}")  # Print the first 5 rows of the DataFrame

    return df


# Usage
df = process_textbook('/Users/rohittiwari/Downloads/ncert_class12_maths_combined.pdf', chunk_size=4, overlap=1)
df.to_csv('processed_ncert_class12_maths.csv', index=False)