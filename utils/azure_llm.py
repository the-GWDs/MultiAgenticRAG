"""
Azure OpenAI LLM integration for the Agentic RAG system.
"""
import logging
from typing import Dict, Any, Optional, List

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

# Azure OpenAI imports
from langchain_azure_openai import AzureChatOpenAI

# Databricks utilities
from utils.databricks_utils import load_databricks_config

# Configure logging
logger = logging.getLogger(__name__)

def get_azure_chat_model(
    model_type: str = "gpt4",
    temperature: float = 0,
    streaming: bool = True,
    config_path: str = "config.yaml"
) -> BaseChatModel:
    """
    Get an Azure OpenAI chat model.
    
    Args:
        model_type (str): The type of model to use ('gpt4' or 'gpt4_mini').
        temperature (float): The temperature to use for generation.
        streaming (bool): Whether to use streaming.
        config_path (str): Path to the configuration file.
        
    Returns:
        BaseChatModel: The Azure OpenAI chat model.
    """
    try:
        # Load configuration
        config = load_databricks_config(config_path)
        
        # Get model deployment name based on type
        if model_type.lower() == "gpt4":
            deployment_name = config["azure_openai"]["deployment_name_gpt4"]
        elif model_type.lower() == "gpt4_mini":
            deployment_name = config["azure_openai"]["deployment_name_gpt4_mini"]
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Create Azure OpenAI chat model
        model = AzureChatOpenAI(
            azure_deployment=deployment_name,
            azure_endpoint=config["azure_openai"]["endpoint"],
            api_version=config["azure_openai"]["api_version"],
            temperature=temperature,
            streaming=streaming
        )
        
        logger.info(f"Created Azure OpenAI chat model with deployment: {deployment_name}")
        return model
    except Exception as e:
        logger.error(f"Error creating Azure OpenAI chat model: {e}")
        raise RuntimeError(f"Error creating Azure OpenAI chat model: {e}")

def get_azure_embeddings(config_path: str = "config.yaml") -> Any:
    """
    Get Azure OpenAI embeddings.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Any: The Azure OpenAI embeddings.
    """
    try:
        # Load configuration
        config = load_databricks_config(config_path)
        
        # Create Azure OpenAI embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config["azure_openai"]["deployment_name_embeddings"],
            azure_endpoint=config["azure_openai"]["endpoint"],
            api_version=config["azure_openai"]["api_version"]
        )
        
        logger.info("Created Azure OpenAI embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating Azure OpenAI embeddings: {e}")
        raise RuntimeError(f"Error creating Azure OpenAI embeddings: {e}")

# Import after function definition to avoid circular import
from langchain_azure_openai import AzureOpenAIEmbeddings 
