"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline
import unicodedata


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    #  Load the CSV and return the DataFrame
    df = pd.read_csv(filepath)
    return df 


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # Compute shape, language/category value_counts, and word-count
    #       statistics on df['text']
    
    shape = df.shape
    #Count occurrences for language and category
    lang_counts = df['language'].value_counts().to_dict()
    category_counts = df['category'].value_counts().to_dict()
    #Calculate word count statistics for text column
    word_counts = df['text'].str.split().str.len()
    text_length_stats = {
        'mean': word_counts.mean(),
        'min': word_counts.min(),
        'max': word_counts.max()
    }
    results = {
        "shape": shape,
        "lang_counts": lang_counts,
        "category_counts": category_counts,
        "text_length_stats": text_length_stats
    }
    return results


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    #  NFC-normalize the text, run it through nlp(), drop
    #       punctuation/whitespace tokens, return lowercased lemmas
    normalized_text=unicodedata.normalize('NFC', text)
    doc = nlp(normalized_text)
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct and not token.is_space
    ]
    
    return tokens
        
def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    #  Filter df to English rows, process each text with nlp,
    #       collect entities into rows, return as a DataFrame
    english_df = df[df['language']=='en']
    entities = []
    for index , row in english_df.iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            entities.append({
              'text_id': row['id'],  
               'entity_text': ent.text,
               'entity_label': ent.label_, 
               'start_char': ent.start_char,
               'end_char': ent.end_char  
            })
    return pd.DataFrame(entities)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    #  Filter df to English rows, run each text through
    #       ner_pipeline, merge ## subword tokens, strip B-/I- prefix
    #       from labels (IOB format), return as a DataFrame
    
    en_df = df[df["language"] == "en"]

    entities = []

    for _, row in en_df.iterrows():
        # Run Hugging Face NER model
        results = ner_pipeline(row["text"])

        # Used to build multi-token entities
        current_entity = None

        for res in results:
            word = res["word"]
            full_label = res["entity"]   # Example: B-ORG or I-ORG
            label = full_label.split("-")[-1]   # Remove B-/I-

            # Start of a new entity
            if full_label.startswith("B-"):

                # Save previous entity before starting a new one
                if current_entity:
                    entities.append(current_entity)

                current_entity = {
                    "text_id": row["id"],
                    "entity_text": word.replace("##", ""),
                    "entity_label": label,
                    "start_char": res["start"],
                    "end_char": res["end"]
                }

            # Continuation of the same entity
            elif full_label.startswith("I-") and current_entity:

                # Merge WordPiece subwords 
                if word.startswith("##"):
                    current_entity["entity_text"] += word.replace("##", "")
                else:
                    # Merge regular continuation tokens
                    current_entity["entity_text"] += " " + word

                # Update end position
                current_entity["end_char"] = res["end"]

        # Add last entity after loop ends
        if current_entity:
            entities.append(current_entity)

    return pd.DataFrame(entities)
            

def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    #  Count entities per label for each system, compute totals,
    #       and derive the three overlap sets by matching on
    #       (text_id, entity_text)
    
    #Count label for both systems
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict()
    hf_counts = hf_df['entity_label'].value_counts().to_dict()
    
    #Create sets of (text_id, entity_text) tuples for comparison
    spacy_set = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    hf_set = set(zip(hf_df['text_id'], hf_df['entity_text']))
    
    # Find overlap and differences using set operations
    both = spacy_set & hf_set  
    spacy_only = spacy_set - hf_set 
    hf_only = hf_set - spacy_set 
    
    return {
      'spacy_counts': spacy_counts,
        'hf_counts': hf_counts,
        'total_spacy': len(spacy_df),
        'total_hf': len(hf_df),
        'both': both,
        'spacy_only': spacy_only,
        'hf_only': hf_only  
    }   


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    #  Match predicted entities to gold entities by text_id +
    #       entity_text + entity_label, compute precision/recall/F1
    
     # Filter predictions to only texts that have gold annotations
    gold_ids = set(gold_df["text_id"])
    predicted_df = predicted_df[
        predicted_df["text_id"].isin(gold_ids)
    ]

    # Match predicted entities to gold entities
    predicted_set = set(zip(
        predicted_df["text_id"],
        predicted_df["entity_text"],
        predicted_df["entity_label"]
    ))
    
       
    gold_set = set(zip(gold_df['text_id'], 
                       gold_df['entity_text'], 
                       gold_df['entity_label']))
    
    # Calculate Metrics using set operations
    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

# Calculate Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1 Score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    
def extract_multilingual_entities(df, multilingual_nlp):
    """Extract named entities from Arabic texts using multilingual spaCy model."""

    ar_df = df[df["language"] == "ar"]

    entities = []

    for _, row in ar_df.iterrows():
        doc = multilingual_nlp(row["text"])

        for ent in doc.ents:
            entities.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(entities)


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")

        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            if hf_metrics is not None:
                print(f"HF evaluation: {hf_metrics}")
                
                
multilingual_nlp = spacy.load("xx_ent_wiki_sm")

ar_entities = extract_multilingual_entities(df, multilingual_nlp)

print(f"Arabic entities: {len(ar_entities)}")  