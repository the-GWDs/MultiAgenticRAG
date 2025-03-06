"""
Databricks-specific utilities for the MultiAgentic RAG system.
"""
import os
import yaml
from typing import Dict, List, Any, Optional
import logging

# Azure imports
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
import azure.openai as azure_openai

# Databricks imports
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient

# LangChain imports
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Configure logging
logger = logging.getLogger(__name__)

def load_databricks_config(config_path: str = "config-databricks.yaml") -> Dict[str, Any]:
    """
    Load the Databricks-specific configuration from the YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise RuntimeError(f"Error loading configuration: {e}")

def get_workspace_client() -> WorkspaceClient:
    """
    Get a Databricks workspace client using the default credential.
    
    Returns:
        WorkspaceClient: The Databricks workspace client.
    """
    try:
        return WorkspaceClient()
    except Exception as e:
        logger.error(f"Error creating Databricks workspace client: {e}")
        raise RuntimeError(f"Error creating Databricks workspace client: {e}")

def get_vector_search_client() -> VectorSearchClient:
    """
    Get a Databricks vector search client using the workspace client.
    
    Returns:
        VectorSearchClient: The Databricks vector search client.
    """
    try:
        workspace_client = get_workspace_client()
        return VectorSearchClient(workspace_client)
    except Exception as e:
        logger.error(f"Error creating Databricks vector search client: {e}")
        raise RuntimeError(f"Error creating Databricks vector search client: {e}")

def get_document_intelligence_client(config: Dict[str, Any]) -> DocumentIntelligenceClient:
    """
    Get an Azure Document Intelligence client.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        
    Returns:
        DocumentIntelligenceClient: The Azure Document Intelligence client.
    """
    try:
        endpoint = config["document_intelligence"]["endpoint"]
        key = config["document_intelligence"]["key"]
        
        # Use key authentication if provided, otherwise use default Azure credential
        if key and key != "YOUR_DOCUMENT_INTELLIGENCE_KEY":
            credential = AzureKeyCredential(key)
        else:
            credential = DefaultAzureCredential()
            
        return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
    except Exception as e:
        logger.error(f"Error creating Document Intelligence client: {e}")
        raise RuntimeError(f"Error creating Document Intelligence client: {e}")

def get_azure_openai_client(config: Dict[str, Any]) -> azure_openai.AzureOpenAI:
    """
    Get an Azure OpenAI client.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        
    Returns:
        azure_openai.AzureOpenAI: The Azure OpenAI client.
    """
    try:
        endpoint = config["azure_openai"]["endpoint"]
        api_version = config["azure_openai"]["api_version"]
        
        # Use default Azure credential for authentication
        credential = DefaultAzureCredential()
            
        return azure_openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_ad_token_provider=credential
        )
    except Exception as e:
        logger.error(f"Error creating Azure OpenAI client: {e}")
        raise RuntimeError(f"Error creating Azure OpenAI client: {e}")

def process_document_with_azure(
    document_path: str, 
    config: Dict[str, Any],
    headers_to_split_on: List[List[str]]
) -> List[Document]:
    """
    Process a document using Azure Document Intelligence and split it into chunks.
    
    Args:
        document_path (str): Path to the document.
        config (Dict[str, Any]): The configuration dictionary.
        headers_to_split_on (List[List[str]]): Headers to split the document on.
        
    Returns:
        List[Document]: List of document chunks.
    """
    try:
        logger.info(f"Processing document: {document_path}")
        
        # Get Document Intelligence client
        client = get_document_intelligence_client(config)
        
        # Read the document
        with open(document_path, "rb") as f:
            document_content = f.read()
        
        # Analyze the document
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            AnalyzeDocumentRequest(document=document_content)
        )
        result = poller.result()
        
        # Extract text content
        content = ""
        for page in result.pages:
            for line in page.lines:
                content += line.content + "\n"
        
        # Split the content using markdown headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        docs = markdown_splitter.split_text(content)
        
        logger.info(f"Document processed successfully, created {len(docs)} chunks")
        return docs
    except Exception as e:
        logger.error(f"Error processing document with Azure: {e}")
        raise RuntimeError(f"Error processing document with Azure: {e}")

def create_vector_search_index(
    config: Dict[str, Any],
    documents: List[Document],
    embeddings_function: Any
) -> None:
    """
    Create a vector search index in Databricks.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        documents (List[Document]): List of documents to index.
        embeddings_function (Any): Function to generate embeddings.
    """
    try:
        logger.info("Creating vector search index in Databricks")
        
        # Get vector search client
        vs_client = get_vector_search_client()
        
        # Get configuration values
        endpoint_name = config["retriever"]["vector_search_endpoint_name"]
        index_name = config["retriever"]["vector_index_name"]
        catalog = config["databricks"]["catalog"]
        schema = config["databricks"]["schema"]
        dimension = config["retriever"]["vector_dimension"]
        
        # Check if endpoint exists, create if not
        endpoints = vs_client.list_endpoints()
        endpoint_exists = any(e.name == endpoint_name for e in endpoints)
        
        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {endpoint_name}")
            vs_client.create_endpoint(
                name=endpoint_name,
                endpoint_type="STANDARD"
            )
        
        # Prepare documents for indexing
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = [embeddings_function.embed_query(text) for text in texts]
        
        # Create or update index
        logger.info(f"Creating vector search index: {index_name}")
        vs_client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=f"{catalog}.{schema}.documents",
            primary_key="id",
            embedding_dimension=dimension,
            embedding_vector_column="embedding"
        )
        
        logger.info("Vector search index created successfully")
    except Exception as e:
        logger.error(f"Error creating vector search index: {e}")
        raise RuntimeError(f"Error creating vector search index: {e}")

def query_vector_search(
    config: Dict[str, Any],
    query: str,
    embeddings_function: Any,
    top_k: int = 3
) -> List[Document]:
    """
    Query the vector search index in Databricks.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        query (str): The query string.
        embeddings_function (Any): Function to generate embeddings.
        top_k (int): Number of results to return.
        
    Returns:
        List[Document]: List of retrieved documents.
    """
    try:
        logger.info(f"Querying vector search index with: {query}")
        
        # Get vector search client
        vs_client = get_vector_search_client()
        
        # Get configuration values
        endpoint_name = config["retriever"]["vector_search_endpoint_name"]
        index_name = config["retriever"]["vector_index_name"]
        
        # Generate query embedding
        query_embedding = embeddings_function.embed_query(query)
        
        # Query the index
        results = vs_client.similarity_search(
            endpoint_name=endpoint_name,
            index_name=index_name,
            query_vector=query_embedding,
            num_results=top_k
        )
        
        # Convert results to Documents
        documents = []
        for result in results:
            doc = Document(
                page_content=result["text"],
                metadata=result["metadata"]
            )
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents from vector search")
        return documents
    except Exception as e:
        logger.error(f"Error querying vector search: {e}")
        raise RuntimeError(f"Error querying vector search: {e}") 
