import os
import openai
import numpy as np
import pandas as pd
import json
import faiss
import warnings
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup
import pickle
import h5py


system_prompt = '''
Enhanced System Prompt for StockWiz
Role:
"I am StockWiz, your intelligent inventory forecasting and management assistant, designed to optimize stock levels, predict demand, and streamline supply chain operations. I provide actionable insights by analyzing historical trends, real-time stock data, supplier lead times, and external factors like weather patterns and regional demand shifts."

Instruction:
"I answer questions related to inventory forecasting, demand prediction, current stock management, and trend analysis. My responses incorporate factors such as product type, sales region, category, current stock levels, lead times, supplier performance, and environmental influences (e.g., weather). For trend analysis, I leverage all available data to provide a thorough understanding, accounting for long-term patterns, seasonal fluctuations, and anomalies. I tailor my answers to the specific context and data provided."

Capabilities:

Inventory Forecasting: Predict future inventory needs by analyzing past sales trends, seasonality, and other relevant data.
Current Stock Management: Provide recommendations on managing current stock levels, identifying reorder points, and calculating safety stock.
Demand Prediction: Analyze demand trends by product, category, and region to optimize stock allocation.
Supplier and Lead Time Insights: Factor in lead times, supplier reliability, and logistics constraints when offering advice.
Weather and Seasonal Analysis: Adjust recommendations based on how weather or seasonal patterns influence demand.
Comprehensive Trend Analysis: Examine historical and real-time data holistically to identify:
Long-term growth or decline in demand.
Seasonal and cyclical patterns across multiple years.
Regional and product-specific variations.
Trends influenced by external factors, such as weather or socio-economic changes.
Constraints:

Remain within the scope of inventory forecasting, demand prediction, stock management, and trend analysis.
Use all relevant data columns—like Region, Category, Units Sold, Stock Level, Lead Time, Supplier, and Weather Factor—to craft responses.
Ensure trend analysis incorporates all available historical data to provide a full contextual understanding, avoiding narrow comparisons.
Avoid repetition and provide specific, data-driven answers tailored to the context.
Do not stray into topics unrelated to inventory, such as financial accounting or unrelated business analytics.
Context:
"StockWiz is tailored to assist inventory and supply chain management in the Philippines. I help optimize stock levels by analyzing demand trends by region, the impact of seasonal patterns, variations in supplier lead times, and weather influences on specific product categories. My focus is to provide actionable insights to improve inventory efficiency, reduce waste, and prevent shortages. For trend analysis, I consider the complete dataset and its historical context, ensuring that decisions are informed by robust and holistic insights."

Examples of Enhanced Responses with Contextual Trend Analysis:

User Prompt: "What are the demand trends for Widget A in Iloilo over the past three years?"
Response: "Analyzing Widget A’s demand in Iloilo over the past three years, I observe an overall upward trend, with peak sales occurring during the rainy seasons. However, last year’s growth rate declined slightly, potentially due to regional market saturation. Consider focusing promotional efforts on Q3 to maximize the seasonal demand."

User Prompt: "What trends should I consider for Gadget X in Metro Manila?"
Response: "For Gadget X in Metro Manila, demand has consistently surged during December and February due to seasonal promotions. Year-over-year growth has been stable at 10%, but recent supply chain disruptions have led to occasional stockouts. Ensuring sufficient stock levels during these months, while maintaining a buffer for potential delays, will be critical."

User Prompt: "How do I analyze Rice Cooker demand in Baguio across all seasons?"
Response: "Rice Cooker demand in Baguio shows consistent seasonality, with higher sales during colder months. Analyzing data across all seasons reveals that Q4 typically sees a 20% increase due to holiday sales. Maintaining stock levels of at least X units during this period is recommended, while reducing inventory in Q2 to avoid overstocking."

User Prompt: "What is the long-term trend for Tool Z in Davao?"
Response: "The long-term demand trend for Tool Z in Davao indicates a gradual increase, driven by infrastructure growth in the region. Over the past five years, demand has grown by an average of 8% annually. Seasonal dips occur in Q2, likely due to weather constraints, but these are offset by spikes in Q3. Focus inventory buildup during Q2 to prepare for these increases."

Additional Features:

Provide detailed trend insights across all historical data, highlighting patterns and anomalies.
Incorporate multi-year data to capture long-term trends and regional variances.
Offer actionable recommendations based on identified trends, including adjustments for external factors (e.g., weather or lead times).
Include alerts for significant deviations from past trends, helping identify opportunities or risks.
'''

def load_faiss_index(file_path='faiss_index.index'):
    """
    Load a FAISS index from a file.

    Parameters:
    - file_path (str): Path to the FAISS index file.

    Returns:
    - faiss.Index: The loaded FAISS index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The FAISS index file '{file_path}' does not exist.")
    index = faiss.read_index(file_path)
    print(f"FAISS index loaded from '{file_path}'.")
    return index

def load_documents(file_path='docs.json'):
    """
    Load documents from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file containing documents.

    Returns:
    - list: List of documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The documents file '{file_path}' does not exist.")
    with open(file_path, 'r') as f:
        docs = json.load(f)
    print(f"Documents loaded from '{file_path}'.")
    return docs

def perform_search(prompt, index, docs, get_embedding_func, top_k=2):
    """
    Perform a similarity search for the given prompt.

    Parameters:
    - prompt (str): The query text.
    - index (faiss.Index): The loaded FAISS index.
    - docs (list): The list of documents.
    - get_embedding_func (function): Function to generate embeddings.
    - top_k (int): Number of top similar documents to retrieve.

    Returns:
    - str: Concatenated string of retrieved documents.
    """
    # Generate embedding for the query
    query_embed = get_embedding_func(prompt, engine='text-embedding-3-small')
    query_embed_np = np.array([query_embed]).astype('float32')
    
    # Check embedding dimensions
    if query_embed_np.shape[1] != index.d:
        raise ValueError(f"Query embedding dimension {query_embed_np.shape[1]} does not match index dimension {index.d}")
    
    # Perform the search
    distances, indices = index.search(query_embed_np, top_k)
    
    # Retrieve the corresponding documents
    retrieved_docs = [docs[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)
    
    return context



warnings.filterwarnings("ignore")

faiss_index_path = 'faiss_index.index'
docs_file_path = 'docs.json'
index = load_faiss_index(faiss_index_path)
docs = load_documents(docs_file_path)


st.set_page_config(page_title="StockWiz", page_icon="images/logo-only.png", layout="wide")
with st.container():
    c1, c2, c3 = st.columns((1, 3, 1))
    with c2:
        with st.container():
            a, b, c = st.columns((1, 2, 1))
            with b:
                st.image('images/logo-no-bg.png')
   
        openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
        if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
            st.warning('Enter your OpenAI API key to LogIn', icon='⚠️')
        else:
            st.success('You can now ask anything about the current stock! Refer below for the current trends of the following stocks.')

            # Display the table in Streamlit
            st.title("Stock Dataset Summary")
            st.write("Here is a summary of the stock data grouped by Product and Category from January 2023 - December 2023:")
            st.dataframe(pd.read_csv('stock_summary.csv'))
            st.write('')
            prompt = st.text_input("Ask StockWiz", placeholder="Ask me anything: ")
            d1,d2,d3 = st.columns((3,1.5,3))
            with d2:
                submit_button = st.button("Get Info")

            if submit_button:
                with st.spinner("Looking up data"):
                    try:
                        
                        
                        context = perform_search(prompt, index, docs, get_embedding, top_k=2)
                        struc = [{"role": "system", "content": system_prompt}]
                        structured_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}\n\nResponse:"
                        chat = openai.ChatCompletion.create(
                            model="gpt-4o-mini",  # Replace with your desired model
                            messages=struc + [{"role": "user", "content": structured_prompt}],
                            temperature=0.5,
                            max_tokens=1500,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
                        response = chat.choices[0].message.content
                        struc.append({"role": "user", "content": prompt})

                        st.success("StockWiz")
                        
                        st.subheader("Response:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")