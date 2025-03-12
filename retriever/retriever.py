"""
Databricks-specific retriever implementation using Azure Document Intelligence and Databricks Vector Search.
"""
import logging
from typing import List, Any, Tuple, Dict, Optional

# LangChain imports
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Azure OpenAI imports
from langchain_azure_openai import AzureOpenAIEmbeddings

# Databricks utilities
from utils.databricks_utils import (
    load_databricks_config,
    process_document_with_azure,
    create_vector_search_index,
    query_vector_search
)

# Configure logging
logger = logging.getLogger(__name__)

class DatabricksIndexBuilder:
    """
    Handles document processing and indexing using Azure Document Intelligence and Databricks Vector Search.
    """
    def __init__(
        self, 
        document_path: str, 
        headers_to_split_on: List[Tuple[str, str]],
        load_documents: bool = False,
        config_path: str = "config.yaml"
    ):
        """
        Initialize the Databricks index builder.
        
        Args:
            document_path (str): Path to the document to process.
            headers_to_split_on (List[Tuple[str, str]]): Headers to split the document on.
            load_documents (bool): Whether to load and process documents.
            config_path (str): Path to the configuration file.
        """
        self.document_path = document_path
        self.headers_to_split_on = headers_to_split_on
        self.load_documents = load_documents
        self.config = load_databricks_config(config_path)
        
        # Initialize components
        self.docs_list = []
        self.vectorstore_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # Process documents if required
        if self.load_documents:
            self.process_documents()
            self.build_vector_search_index()
        
        # Build retrievers
        self.build_retrievers()
    
    def process_documents(self) -> None:
        """
        Process documents using Azure Document Intelligence.
        """
        try:
            logger.info(f"Processing document: {self.document_path}")
            self.docs_list = process_document_with_azure(
                document_path=self.document_path,
                config=self.config,
                headers_to_split_on=self.headers_to_split_on
            )
            logger.info(f"Document processed successfully, created {len(self.docs_list)} chunks")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise RuntimeError(f"Error processing document: {e}")
    
    def build_vector_search_index(self) -> None:
        """
        Build a vector search index in Databricks.
        """
        try:
            logger.info("Building vector search index")
            
            # Create Azure OpenAI embeddings
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.config["azure_openai"]["deployment_name_embeddings"],
                azure_endpoint=self.config["azure_openai"]["endpoint"],
                api_version=self.config["azure_openai"]["api_version"]
            )
            
            # Create vector search index
            create_vector_search_index(
                config=self.config,
                documents=self.docs_list,
                embeddings_function=embeddings
            )
            
            logger.info("Vector search index built successfully")
        except Exception as e:
            logger.error(f"Error building vector search index: {e}")
            raise RuntimeError(f"Error building vector search index: {e}")
    
    def build_retrievers(self) -> None:
        """
        Build retrievers for document retrieval.
        """
        try:
            logger.info("Building retrievers")
            
            # Create Azure OpenAI embeddings
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.config["azure_openai"]["deployment_name_embeddings"],
                azure_endpoint=self.config["azure_openai"]["endpoint"],
                api_version=self.config["azure_openai"]["api_version"]
            )
            
            # Create vector search retriever
            self.vectorstore_retriever = self._create_vector_search_retriever(embeddings)
            
            # Create BM25 retriever if documents are available
            if self.docs_list:
                self.bm25_retriever = BM25Retriever.from_documents(self.docs_list)
                self.bm25_retriever.k = self.config["retriever"]["top_k"]
            
            # Create ensemble retriever if both retrievers are available
            if self.vectorstore_retriever and self.bm25_retriever:
                weights = self.config["retriever"]["ensemble_weights"]
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.vectorstore_retriever, self.bm25_retriever],
                    weights=weights[:2]  # Use only the first two weights for these retrievers
                )
                
                # Create compression retriever with Cohere reranking
                self._create_compression_retriever()
            
            logger.info("Retrievers built successfully")
        except Exception as e:
            logger.error(f"Error building retrievers: {e}")
            raise RuntimeError(f"Error building retrievers: {e}")
    
    def _create_vector_search_retriever(self, embeddings: Any) -> Any:
        """
        Create a retriever that uses Databricks Vector Search.
        
        Args:
            embeddings (Any): The embeddings function to use.
            
        Returns:
            Any: The vector search retriever.
        """
        class DatabricksVectorSearchRetriever:
            def __init__(self, config, embeddings_function, top_k=3):
                self.config = config
                self.embeddings_function = embeddings_function
                self.top_k = top_k
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                return query_vector_search(
                    config=self.config,
                    query=query,
                    embeddings_function=self.embeddings_function,
                    top_k=self.top_k
                )
        
        return DatabricksVectorSearchRetriever(
            config=self.config,
            embeddings_function=embeddings,
            top_k=self.config["retriever"]["top_k"]
        )
    
    def _create_compression_retriever(self) -> None:
        """
        Create a compression retriever with Cohere reranking.
        """
        try:
            # Create Cohere reranker
            compressor = CohereRerank(
                model=self.config["retriever"]["cohere_rerank_model"],
                top_n=self.config["retriever"]["top_k_compression"]
            )
            
            # Create compression retriever
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.ensemble_retriever
            )
            
            logger.info("Compression retriever created successfully")
        except Exception as e:
            logger.error(f"Error creating compression retriever: {e}")
            raise RuntimeError(f"Error creating compression retriever: {e}")
    
    def get_retriever(self) -> Any:
        """
        Get the best available retriever.
        
        Returns:
            Any: The retriever to use.
        """
        if hasattr(self, "compression_retriever") and self.compression_retriever:
            return self.compression_retriever
        elif self.ensemble_retriever:
            return self.ensemble_retriever
        elif self.vectorstore_retriever:
            return self.vectorstore_retriever
        else:
            raise RuntimeError("No retriever available") 
