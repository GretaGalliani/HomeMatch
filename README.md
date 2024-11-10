# AI-Powered Property Matching System

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-darkgreen.svg)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-orange.svg)](https://www.langchain.com/)

## Project Overview

This project implements an intelligent property matching system that leverages Large Language Models (LLMs) and embedding-based similarity search to connect home buyers with their ideal properties. Built as part of the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608?promo=oct&coupon=BOOST40&utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=158023437317&utm_keyword=udacity%20generative%20ai_e&utm_source=gsem_brand&utm_medium=ads_r&utm_campaign=19167921312_c_individuals&utm_term=158023437317&utm_keyword=udacity%20generative%20ai_e&gad_source=1&gbraid=0AAAAADmkdSQJCv4xuXntYTnJL6NSmME3x&gclid=Cj0KCQiA0MG5BhD1ARIsAEcZtwTxE7Ch68XsTIsdV90PKGAIfhCfQyVlbBX2_rrjpulOwvFQ5aPcjwUaAqppEALw_wcB), the system demonstrates practical applications of modern AI techniques in the real estate domain.

### Key Features

- 🏠 Synthetic property listing generation using LLMs
- 🔍 Semantic search using embedding-based similarity
- 🎯 Personalized property recommendations
- 💡 AI-powered listing refinement based on user preferences
- 🖥️ Interactive web interface for preference collection

## Technical Architecture

The system consists of two main components:

### 1. Synthetic Data Generator (`SyntheticDataGenerator.ipynb`)

- Generates realistic property listings using OpenAI's GPT models
- Computes embeddings for efficient similarity search
- Stores listings and embeddings in ChromaDB for quick retrieval
- Handles data preprocessing and augmentation

### 2. Property Matcher (`HomeMatch.ipynb`)

- Collects user preferences through an interactive form
- Performs semantic search using embedding similarity
- Fine-tunes property descriptions based on user preferences
- Presents personalized property recommendations
- Implements a user-friendly interface using IPython widgets

## Installation

```bash
# Clone the repository
git clone https://github.com/GretaGalliani/HomeMatch.git
cd cd HomeMatch

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-base-url"
```

## Usage

### 1. Generate Synthetic Listings
Run notebook `SyntheticDataGenerator.ipynb`

### 2. Match Properties with User Preferences
Run notebook `HomeMatch.ipynb`

## Technologies Used

- **Python 3.12+**: Core programming language
- **OpenAI API**: For LLM-based text generation and embeddings
- **LangChain**: Framework for LLM application development
- **ChromaDB**: Vector database for similarity search
- **IPython Widgets**: Interactive user interface components
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Pydantic**: Data validation and settings management

---

*This project was developed as part of the Udacity Generative AI Nanodegree program and serves as a demonstration of applying modern AI techniques to real-world problems.*
