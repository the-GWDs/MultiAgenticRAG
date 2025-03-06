"""
Databricks-specific version of the MultiAgentic RAG application.
"""
from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import new_uuid
from subgraph.graph_builder import researcher_graph
from main_graph.graph_builder import InputState, graph
from langgraph.types import Command
import asyncio
import uuid
import time
import builtins
import logging

# Databricks-specific imports
from utils.databricks_utils import load_databricks_config
from retriever.databricks_retriever import DatabricksIndexBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_databricks_config()

# Initialize the index builder
def initialize_index_builder():
    """
    Initialize the Databricks index builder.
    
    Returns:
        DatabricksIndexBuilder: The initialized index builder.
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

# Initialize the researcher graph
researcher = researcher_graph.compile()

async def process_query(query: str):
    """
    Process a query using the MultiAgentic RAG system.
    
    Args:
        query (str): The query to process.
        
    Returns:
        str: The response to the query.
    """
    try:
        logger.info(f"Processing query: {query}")
        
        # Create a unique ID for the query
        query_id = new_uuid()
        
        # Create the input state
        input_state = InputState(
            query=query,
            query_id=query_id
        )
        
        # Initialize the agent state
        agent_state = AgentState(
            messages=[{"role": "user", "content": query}],
            query=query,
            query_id=query_id,
            research_plan=[],
            research_summary="",
            hallucination_score=0,
            hallucination_explanation="",
            docs=[]
        )
        
        # Initialize the main graph
        main_graph = graph.compile()
        
        # Process the query
        result = await main_graph.ainvoke(agent_state)
        
        # Extract the response
        response = result.messages[-1]["content"]
        
        logger.info("Query processed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing query: {e}"

# Example usage in a Databricks notebook
if __name__ == "__main__":
    # This code would typically be run in a Databricks notebook
    query = "What are Google's carbon emissions goals?"
    response = asyncio.run(process_query(query))
    print(response)
    
    # In a Databricks notebook, you might use:
    # display(response) 
