#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re


# In[2]:


# Download NLTK data
nltk.download('punkt')


# In[6]:


# Function to read stop words
def load_stop_words(directory):
    stop_words = set()
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    for file_name in os.listdir(directory):
        for encoding in encodings:
            try:
                with open(os.path.join(directory, file_name), 'r', encoding=encoding) as file:
                    stop_words.update(file.read().splitlines())
                break  
            except UnicodeDecodeError:
                continue  
    return stop_words

# Function to read positive/negative words
def load_words(file_path):
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                words = set(file.read().splitlines())
            return words  
        except UnicodeDecodeError:
            continue  

# Load stop words, positive, and negative words
stop_words = load_stop_words('Desktop/StopWords')
positive_words = load_words('Desktop/MasterDictionary/positive-words.txt')
negative_words = load_words('Desktop/MasterDictionary/negative-words.txt')


# In[9]:


# Read the Excel file containing URLs and their IDs
excel_file = 'Downloads/Input.xlsx'
df = pd.read_excel(excel_file)


# In[10]:


urls = df['URL']
ids=df['URL_ID']


# In[11]:


# Create a directory to save the text files
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[12]:


# Common selectors for article content
article_selectors = [
    {'tag': 'article'},
    {'tag': 'div', 'class': 'content'},
    {'tag': 'div', 'class': 'post'},
    {'tag': 'div', 'class': 'entry-content'},
    {'tag': 'section', 'class': 'content'},
    {'tag': 'div', 'id': 'content'},
]


# In[14]:


# Function to fetch content from a URL
def fetch_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        article_text = None
        for selector in article_selectors:
            tag = selector.get('tag')
            class_name = selector.get('class')
            id_name = selector.get('id')

            if class_name:
                article_content = soup.find(tag, class_=class_name)
            elif id_name:
                article_content = soup.find(tag, id=id_name)
            else:
                article_content = soup.find(tag)

            if article_content:
                paragraphs = article_content.find_all('p')
                article_text = '\n'.join([p.get_text() for p in paragraphs])
                break

        if not article_text:
            article_text = "Article content not found."
        return article_text
    except Exception as e:
        return f"Failed to fetch content from {url}: {e}"

# Function to clean text
def clean_text(text):
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return cleaned_tokens

# Function to calculate syllables in a word
def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    if word[0] in vowels:
        syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllables += 1
    if word.endswith("e"):
        syllables -= 1
    if word.endswith("es") or word.endswith("ed"):
        syllables -= 1
    if syllables == 0:
        syllables += 1
    return syllables

# Function to compute textual analysis
def compute_textual_analysis(text):
    # Clean the text
    cleaned_tokens = clean_text(text)
    cleaned_text = ' '.join(cleaned_tokens)
    
    # Tokenize into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    cleaned_words = word_tokenize(cleaned_text)

    # Calculate positive and negative scores
    positive_score = sum(1 for word in cleaned_words if word in positive_words)
    negative_score = sum(1 for word in cleaned_words if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(cleaned_words) + 0.000001)

    # Calculate readability and other metrics
    avg_sentence_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)
    complex_word_count = sum(1 for word in cleaned_words if count_syllables(word) > 2)
    percentage_complex_words = complex_word_count / len(cleaned_words) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = len(words) / len(sentences)
    word_count = len(cleaned_words)
    syllable_per_word = sum(count_syllables(word) for word in cleaned_words) / len(cleaned_words)
    personal_pronouns = len(re.findall(r'\b(I|we|my|ours|us)\b', text, re.I))
    avg_word_length = sum(len(word) for word in cleaned_words) / len(cleaned_words)

    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllable_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

# Create a DataFrame for the output structure
output_df = pd.DataFrame(columns=[
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE',
    'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
    'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
])

# Iterate over the URLs and save the content
for url, url_id in zip(urls, ids):
    article_text = fetch_content(url)
    
    if article_text:
        analysis_results = compute_textual_analysis(article_text)
        output_row = {'URL_ID': url_id, 'URL': url}
        output_row.update(analysis_results)
        output_df = output_df.append(output_row, ignore_index=True)

# Save the output DataFrame to an Excel file
output_file = 'output_analysis.xlsx'
output_df.to_excel(output_file, index=False)
print(f'Textual analysis results saved to {output_file}')


# In[ ]:




