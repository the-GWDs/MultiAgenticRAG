# Databricks notebook source
# MAGIC %md
# MAGIC # MultiAgentic RAG System on Databricks
# MAGIC 
# MAGIC This notebook demonstrates how to run the MultiAgentic RAG system on Databricks using Azure services.
# MAGIC 
# MAGIC ## Setup
# MAGIC 
# MAGIC First, we need to install the required dependencies:

# COMMAND ----------

# MAGIC %pip install -r requirements-databricks.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC Next, we need to configure the system with the appropriate Azure and Databricks settings:

# COMMAND ----------

import yaml
import os

# Define configuration
config = {
    "retriever": {
        "file": "retriever/google-2024-environmental-report.pdf",
        "headers_to_split_on": [
            ["#", "Header 1"],
            ["##", "Header 2"]
        ],
        "load_documents": True,
        "vector_search_endpoint_name": "rag_vector_search",
        "vector_index_name": "rag_vector_index",
        "vector_dimension": 1536,
        "top_k": 3,
        "top_k_compression": 3,
        "ensemble_weights": [0.3, 0.3, 0.4],
        "cohere_rerank_model": "rerank-english-v3.0"
    },
    "document_intelligence": {
        "endpoint": dbutils.secrets.get("rag-scope", "document-intelligence-endpoint"),
        "key": dbutils.secrets.get("rag-scope", "document-intelligence-key")
    },
    "azure_openai": {
        "api_version": "2023-05-15",
        "endpoint": dbutils.secrets.get("rag-scope", "azure-openai-endpoint"),
        "deployment_name_gpt4": "gpt-4o",
        "deployment_name_gpt4_mini": "gpt-4o-mini",
        "deployment_name_embeddings": "text-embedding-ada-002",
        "temperature": 0
    },
    "databricks": {
        "workspace_url": dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get(),
        "catalog": "rag_catalog",
        "schema": "rag_schema"
    }
}

# Save configuration to file
with open("config-databricks.yaml", "w") as f:
    yaml.dump(config, f)

print("Configuration saved to config-databricks.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schema
# MAGIC 
# MAGIC Now, let's create the catalog and schema for storing our vector data:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create catalog if it doesn't exist
# MAGIC CREATE CATALOG IF NOT EXISTS rag_catalog;
# MAGIC 
# MAGIC -- Use the catalog
# MAGIC USE CATALOG rag_catalog;
# MAGIC 
# MAGIC -- Create schema if it doesn't exist
# MAGIC CREATE SCHEMA IF NOT EXISTS rag_schema;
# MAGIC 
# MAGIC -- Use the schema
# MAGIC USE SCHEMA rag_schema;
# MAGIC 
# MAGIC -- Create documents table if it doesn't exist
# MAGIC CREATE TABLE IF NOT EXISTS documents (
# MAGIC   id STRING,
# MAGIC   text STRING,
# MAGIC   metadata MAP<STRING, STRING>,
# MAGIC   embedding ARRAY<FLOAT>
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize the System
# MAGIC 
# MAGIC Now, let's initialize the MultiAgentic RAG system:

# COMMAND ----------

import logging
import asyncio
from retriever.databricks_retriever import DatabricksIndexBuilder
from utils.databricks_utils import load_databricks_config
from utils.azure_llm import get_azure_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_databricks_config()

# Initialize embeddings
embeddings = get_azure_embeddings()

# Initialize the index builder
def initialize_index_builder():
    """
    Initialize the Databricks index builder.
    """
    try:
        logger.info("Initializing Databricks index builder")
        
        # Get configuration values
        document_path = config["retriever"]["file"]
        headers_to_split_on = config["retriever"]["headers_to_split_on"]
        load_documents = config["retriever"]["load_documents"]
        
        # Initialize index builder
        index_builder = DatabricksIndexBuilder(
            document_path=document_path,
            headers_to_split_on=headers_to_split_on,
            load_documents=load_documents
        )
        
        logger.info("Databricks index builder initialized successfully")
        return index_builder
    except Exception as e:
        logger.error(f"Error initializing Databricks index builder: {e}")
        raise RuntimeError(f"Error initializing Databricks index builder: {e}")

# Initialize the index builder
index_builder = initialize_index_builder()

# Get the retriever
retriever = index_builder.get_retriever()

print("System initialized successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the MultiAgentic RAG System
# MAGIC 
# MAGIC Now, let's run the system with a sample query:

# COMMAND ----------

from app_databricks import process_query

# Define a query
query = "What are Google's carbon emissions goals?"

# Process the query
response = asyncio.run(process_query(query))

# Display the response
print("Query:", query)
print("\nResponse:")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Query Interface
# MAGIC 
# MAGIC Let's create a simple interface for querying the system:

# COMMAND ----------

# Create a text input widget
dbutils.widgets.text("query", "What are Google's carbon emissions goals?", "Enter your query")

# Get the query from the widget
query = dbutils.widgets.get("query")

# Process the query
if query:
    response = asyncio.run(process_query(query))
    
    # Display the response
    displayHTML(f"<h3>Query:</h3><p>{query}</p><h3>Response:</h3><p>{response}</p>")
else:
    print("Please enter a query in the text input widget above.") 
