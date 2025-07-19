#!/usr/bin/env python
# coding: utf-8

# In[62]:


import nltk
import re
import math
import random
from collections import defaultdict, Counter
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


# In[63]:


nltk.download('gutenberg', quiet=True)
nltk.download('punkt', quiet=True)


# In[64]:


# Collecting and cleaning the data


# In[65]:


raw_data = " ".join(gutenberg.raw(file_id) for file_id in gutenberg.fileids())
print(f"Raw data length: {len(raw_data)} characters")


# In[66]:


def preprocess_data(data):
    """Clean and preprocess text data."""
    data = data.lower()
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    return data

cleaned_text = preprocess_data(raw_data)
print("Data is cleaned and preprocessed")


# In[67]:


# Tokenization
tokens = word_tokenize(cleaned_text)
print(f"Total tokens: {len(tokens)}")
print(f"Sample tokens: {tokens[:10]}")


# In[68]:


# Spliting the data into train/test sets (90/10)
split_point = int(0.9 * len(tokens))
train_tokens = tokens[:split_point]
test_tokens = tokens[split_point:]
print(f"Train tokens: {len(train_tokens)}, Test tokens: {len(test_tokens)}")


# In[69]:


# Vocab
vocab = set(train_tokens)
V = len(vocab)
print(f"Vocabulary size: {V}")


# In[70]:


# N-gram Model


# In[71]:


class NGramModel:
    
    def __init__(self, n, train_tokens):
        self.n = n
        self.train_tokens = train_tokens
        self.vocab = set(train_tokens)
        self.V = len(self.vocab)
        
    
        self.ngram_freq = defaultdict(int)
        self.context_freq = defaultdict(int)
        
        for ngram in ngrams(train_tokens, n):
            self.ngram_freq[ngram] += 1
            context = ngram[:-1]
            self.context_freq[context] += 1
    
    def get_probability(self, ngram):
        context = ngram[:-1]
        numerator = self.ngram_freq[ngram] + 1
        denominator = self.context_freq[context] + self.V
        return numerator / denominator
    
    def predict_next_word(self, context):
        if len(context) != self.n - 1:
            raise ValueError(f"Context must have exactly {self.n-1} words for {self.n}-gram model")
        
        candidates = []
        context_tuple = tuple(context)
        
        for word in self.vocab:
            ngram = context_tuple + (word,)
            prob = self.get_probability(ngram)
            candidates.append((word, prob))
        
        if not candidates:
            return random.choice(list(self.vocab))
        
        return max(candidates, key=lambda x: x[1])[0]
    
    def generate_sentence(self, start_words, length=10):
        if isinstance(start_words, str):
            start_words = [start_words]
        
        
        if len(start_words) < self.n - 1:
            # Padding with most common words if required
            common_words = [word for word, _ in Counter(self.train_tokens).most_common(self.n - 1)]
            start_words = (common_words[:self.n - 1 - len(start_words)] + start_words)
        elif len(start_words) > self.n - 1:
            start_words = start_words[-(self.n - 1):]
        
        sentence = list(start_words)
        
        for _ in range(length - len(start_words)):
            context = sentence[-(self.n - 1):]
            next_word = self.predict_next_word(context)
            sentence.append(next_word)
        
        return ' '.join(sentence)
    
    def calculate_perplexity(self, test_tokens):
        """Calculating perplexity on test data."""
        log_prob_sum = 0
        N = len(test_tokens)
        
        for i in range(self.n - 1, N):
            ngram = tuple(test_tokens[i - self.n + 1:i + 1])
            prob = self.get_probability(ngram)
            log_prob_sum += math.log(prob)
            
        N = N - self.n + 1
        perplexity = math.exp(-log_prob_sum / N)
        return perplexity


# In[72]:


# Creating models for different n values in the range of 2 to 5
print("Building n-gram models...")
models = {}
for n in [2, 3, 4, 5]:
    print(f"Building {n}-gram model...")
    models[n] = NGramModel(n, train_tokens)


# In[73]:


# 3a. Testing
print("3a. Testing text generation with various prefixes:")
test_prefixes = [
    ["he"],
    ["i"],
    ["the"],
    ["she"],
    ["and"],
    ["it", "was"],
    ["of", "the"],
    ["in", "the"]
]

for n in [2, 3, 4]:  
    print(f"\n {n}-gram Model Results")
    for prefix in test_prefixes:
        if len(prefix) <= n - 1: 
            try:
                generated = models[n].generate_sentence(prefix, length=10)
                print(f"Prefix {prefix}: {generated}")
            except Exception as e:
                print(f"Error with prefix {prefix}: {e}")

# Calculating perplexity 
print("\n Perplexity Evaluation:")
perplexities = {}
for n in [2, 3, 4, 5]:
    try:
        perplexity = models[n].calculate_perplexity(test_tokens)
        perplexities[n] = perplexity
        print(f"{n}-gram Perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error calculating perplexity for {n}-gram: {e}")


best_model = min(perplexities, key=perplexities.get)
print(f"\nBest performing model is: {best_model}-gram (perplexity: {perplexities[best_model]:.2f})")


# In[74]:


def generate_sentence(start_word, length=10):
    
    return models[2].generate_sentence(start_word, length)

def generate_trigram_sentence(start_word, length=10):
    
    return models[3].generate_sentence(start_word, length)

def generate_fourgram_sentence(start_word, length=10):
    
    return models[4].generate_sentence(start_word, length)

def generate_fivegram_sentence(start_word, length=10):
    
    return models[5].generate_sentence(start_word, length)

# Tests
if __name__ == "__main__":
    
    try:
        print("Testing generate_sentence:", generate_sentence("the", 8))
        print("Testing generate_trigram_sentence:", generate_trigram_sentence("the", 8))
        print("Testing generate_fourgram_sentence:", generate_fourgram_sentence("the", 8))
        print("Testing generate_fivegram_sentence:", generate_fivegram_sentence("the", 8))
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")

