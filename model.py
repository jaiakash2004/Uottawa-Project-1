# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fitz
from docx import Document

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

# Function to convert text to a Word document using python-docx
def convert_to_docx(text, docx_path):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(docx_path)

# Function to extract the abstract from a PDF file based on a specific pattern
def extract_abstract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    abstract = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()

        if page_num == 0:
            abstract = text.split("Keywords")[0].strip()

    doc.close()
    return abstract

# Function to prepare text data for inference
def make_inference_dataset(texts):
    texts = tf.convert_to_tensor(texts)
    texts = text_vectorizer(texts)
    return texts

# Function to perform inference using a trained model
def perform_inference(model, abstracts):
    inference_dataset = make_inference_dataset(abstracts)
    predicted_probabilities = model.predict(inference_dataset)

    for i, text in enumerate(abstracts):
        predicted_labels = [label for _, label in sorted(zip(predicted_probabilities[i], vocab), reverse=True)][:3]
        predicted_labels_string = ", ".join(predicted_labels)

        print(f"Abstract: {text}")
        print(f"Top Predicted Labels: {predicted_labels_string}")
        print()

# Load arXiv data from a CSV file hosted on GitHub
arxiv_data = pd.read_csv("https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv")
arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]

# Specify the path of the saved model
saved_model_path = "C:/Users/lenovo/Desktop/Research/TASK 1/my_model"
# Load the pre-trained model
loaded_model = load_model(saved_model_path)

# Process terms data for multi-hot encoding
terms = tf.ragged.constant(arxiv_data["terms"].apply(literal_eval).values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()

# Set up vocabulary and text vectorization
vocabulary = set()
arxiv_data["summaries"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
text_vectorizer = layers.TextVectorization(max_tokens=153375, ngrams=2, output_mode="tf_idf")
text_vectorizer.adapt(arxiv_data["summaries"])

# Specify the path of the PDF file for processing
pdf_path = "C:/Users/lenovo/Desktop/Research/TASK 1/test2.pdf"
# Extract abstract from the PDF and perform inference
abstract_text = extract_abstract_from_pdf(pdf_path)
perform_inference(loaded_model, [abstract_text])

# Specify the path for the output Word document
docx_path = "C:/Users/lenovo/Desktop/Research/TASK 1/output/document.docx"
# Extract text from the entire PDF and convert it to a Word document
pdf_text = extract_text_from_pdf(pdf_path)
convert_to_docx(pdf_text, docx_path)
