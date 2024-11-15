import os
import openai
import numpy as np
import pandas as pd
import json
import faiss
import warnings
import openai
from openai.embeddings_utils import get_embedding
import requests
from bs4 import BeautifulSoup
import pickle
import h5py

def process_csv_files(directory):
    """
    Enhanced function to process all CSV files in the specified directory.
    Combines data into a single DataFrame with a 'combined' column for embeddings.

    Parameters:
    - directory (str): Path to the directory containing CSV files.

    Returns:
    - pd.DataFrame: Combined DataFrame with a 'combined' column.
    """
    all_data = []
    csv_files = [file for file in os.listdir(directory) if file.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in directory '{directory}'.")
        return pd.DataFrame()
    
    # Read all CSV files and collect their columns
    dataframes = []
    all_columns = set()
    for file in csv_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_csv(file_path)
            df = df.fillna('N/A')  # Handle missing values
            dataframes.append((file, df))
            all_columns.update(df.columns)
            print(f"Successfully read '{file_path}' with columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")
    
    # Standardize columns across all DataFrames
    standardized_dfs = []
    for file, df in dataframes:
        # Add missing columns with 'N/A'
        for col in all_columns:
            if col not in df.columns:
                df[col] = 'N/A'
        # Reorder columns to match the unified column list
        df = df[list(all_columns)]
        standardized_dfs.append(df)
    
    # Combine all DataFrames into one
    combined_df = pd.concat(standardized_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Create 'combined' column by concatenating all column values
    combined_df['combined'] = combined_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    return combined_df[['combined']]  # Return only the 'combined' column

def save_embeddings(embeddings, method='npy', file_path='embeddings.npy'):
    """
    Save embeddings to a file using the specified method.

    Parameters:
    - embeddings (np.ndarray or list): Embeddings to save.
    - method (str): The method to use for saving ('npy', 'npz', 'pickle', 'json', 'txt', 'h5').
    - file_path (str): The path where the embeddings will be saved.

    Returns:
    - None
    """
    if method == 'npy':
        # Save as NumPy binary file
        np.save(file_path, embeddings)
        print(f"Embeddings saved as NumPy binary file to '{file_path}'.")
    
    elif method == 'npz':
        # Save multiple arrays in a compressed .npz file
        np.savez_compressed(file_path, embeddings=embeddings)
        print(f"Embeddings saved as compressed NumPy file to '{file_path}'.")
    
    elif method == 'pickle':
        # Save using Pickle
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved using Pickle to '{file_path}'.")
    
    elif method == 'json':
        # Save as JSON
        # Ensure embeddings are serializable (e.g., convert NumPy arrays to lists)
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        with open(file_path, 'w') as f:
            json.dump(embeddings, f)
        print(f"Embeddings saved as JSON to '{file_path}'.")
    
    elif method == 'txt':
        # Save as plain text, one embedding per line
        with open(file_path, 'w') as f:
            for emb in embeddings:
                emb_str = ' '.join(map(str, emb))
                f.write(f"{emb_str}\n")
        print(f"Embeddings saved as plain text to '{file_path}'.")
    
    elif method == 'h5':
        # Save using HDF5 format
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data=embeddings)
        print(f"Embeddings saved in HDF5 format to '{file_path}'.")
    
    else:
        print(f"Unsupported save method '{method}'. Please choose from 'npy', 'npz', 'pickle', 'json', 'txt', 'h5'.")

def save_faiss_index(index, file_path='faiss_index.index'):
    """
    Save a FAISS index to a file.

    Parameters:
    - index (faiss.Index): The FAISS index to save.
    - file_path (str): The path where the index will be saved.

    Returns:
    - None
    """
    faiss.write_index(index, file_path)
    print(f"FAISS index saved to '{file_path}'.")

def save_documents(docs, file_path='docs.json'):
    """
    Save the list of documents to a JSON file.

    Parameters:
    - docs (list): List of document strings.
    - file_path (str): The path where the documents will be saved.

    Returns:
    - None
    """
    with open(file_path, 'w') as f:
        json.dump(docs, f)
    print(f"Documents saved to '{file_path}'.")

directory_path = '.'  # Current directory; change as needed
df = pd.read_csv('stock.csv')
df2 = pd.read_csv('stock_summary.csv')

combined_df = pd.concat([df, df2], ignore_index=True)
combined_df.fillna("")

if df.empty:
    print("No data to process.")

# Step 2: Create 'combined' column
combined_df['combined'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Path to the keys.json file
config_path = os.path.join("config", "keys.json")

# Load the JSON data
with open(config_path, "r") as file:
    keys = json.load(file)
openai.api_key = keys.get("API_KEY")

# Step 3: Generate embeddings
docs = combined_df['combined'].tolist()
docs = [doc for doc in docs if isinstance(doc, str) and len(doc.strip()) > 0]
embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in docs]

if not embeddings:
    print("No embeddings were generated.")

embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype('float32')
print(f"Generated embeddings with dimension: {embedding_dim}")

# Step 4: Save embeddings
save_method = 'npy'  # Choose from 'npy', 'npz', 'pickle', 'json', 'txt', 'h5'
embeddings_save_path = 'embeddings.npy'  # Change file extension based on method
save_embeddings(embeddings_np, method=save_method, file_path=embeddings_save_path)

# Step 5: Save documents
docs_save_path = 'docs.json'
save_documents(docs, file_path=docs_save_path)

# Step 6: Build FAISS index
index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance; choose appropriate index type
index.add(embeddings_np)
print(f"FAISS index built with {index.ntotal} vectors.")

# Step 7: Save FAISS index
faiss_save_path = 'faiss_index.index'
save_faiss_index(index, file_path=faiss_save_path)
