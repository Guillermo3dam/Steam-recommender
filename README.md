# 🎮 Steam Game Recommender System

A hybrid machine learning recommendation system that suggests video games based on both structured filters (genres, tags, categories) and semantic similarity using NLP embeddings.

---

## 🚀 Project Goal

The goal of this project is to build an intelligent recommendation system for Steam games that can understand user preferences in two ways:

- Explicitly (filters like genres and tags)  
- Semantically (natural language descriptions)

---

## 🧠 How it works

This system combines:

- Data filtering (genres, tags, categories, developers)  
- Sentence Transformer embeddings for text understanding  
- FAISS for fast similarity search  
- Ranking system based on similarity scores  

---

## ⚙️ Pipeline

- Data cleaning and preprocessing  
- Feature engineering (tags, genres, publishers)  
- Text embedding generation using Sentence Transformers  
- FAISS index creation for vector search  
- Hybrid recommendation logic (filters + embeddings)  
- Streamlit web application for user interaction  

---

## 💡 Key Features

- 🎯 Hybrid recommendation system (filters + NLP)  
- 🧠 Semantic search using Sentence Transformers  
- ⚡ Fast similarity search with FAISS  
- 🎮 Steam games dataset processing  
- 🌐 Interactive Streamlit interface  

---

## 🛠 Tech Stack

- Python  
- Pandas / NumPy  
- Scikit-learn  
- PyTorch  
- SentenceTransformers  
- FAISS  
- Streamlit  
- Jupyter Notebooks  

---

## 📂 Project Structure

```text
app/        → Streamlit application
src/        → Core recommendation logic
notebooks/  → Data analysis and experiments
data/       → Raw and processed datasets
```

## 👨‍💻 My Contributions

- Designed hybrid recommendation architecture
- Implemented embedding-based similarity search
- Built FAISS indexing pipeline
- Developed Streamlit interactive UI
- Performed data cleaning and feature engineering

---

## 📊 Dataset

Steam video game dataset including:

- Game descriptions
- Genres and tags
- Developers and publishers
- Estimated owners

---

## 🚀 Impact

This project demonstrates how combining traditional filtering techniques with modern NLP embeddings can significantly improve recommendation quality, allowing users to discover games in a more intelligent and personalized way.

---

## 📫 Contact

Guillermo Arévalo Mantilla
📧 guillermo3.dam@gmail.com
