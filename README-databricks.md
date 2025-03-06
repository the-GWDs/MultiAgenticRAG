# MultiAgentic RAG on Databricks

This branch contains a Databricks-specific implementation of the MultiAgentic RAG system. It leverages Databricks Vector Search, Azure Document Intelligence, and Azure OpenAI to provide a powerful RAG system that can be deployed in a Databricks workspace with limited internet access.

## Architecture

The Databricks implementation replaces key components of the original system:

1. **Chroma Vector Database** → **Databricks Vector Search**
   - Uses Databricks' native vector search capabilities for efficient similarity search
   - Stores embeddings in Delta tables for persistence and scalability

2. **Document Processing** → **Azure Document Intelligence**
   - Uses Azure Document Intelligence for advanced document parsing and text extraction
   - Maintains the same document chunking strategy based on headers

3. **OpenAI LLM** → **Azure OpenAI**
   - Uses Azure OpenAI for embeddings and text generation
   - Maintains the same model capabilities through Azure deployments

## Setup Instructions

### 1. Prerequisites

- A Databricks workspace with Unity Catalog enabled
- Azure OpenAI service with deployed models:
  - GPT-4o (or equivalent)
  - GPT-4o-mini (or equivalent)
  - text-embedding-ada-002 (or equivalent)
- Azure Document Intelligence service

### 2. Secret Configuration

Create a secret scope in your Databricks workspace called `rag-scope` with the following secrets:

- `azure-openai-endpoint`: Your Azure OpenAI endpoint URL
- `document-intelligence-endpoint`: Your Azure Document Intelligence endpoint URL
- `document-intelligence-key`: Your Azure Document Intelligence API key

### 3. Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/MultiAgenticRAG.git
   cd MultiAgenticRAG
   git checkout dbx
   ```

2. Upload the repository to your Databricks workspace using the Databricks CLI or the UI.

3. Install the required dependencies by running:
   ```bash
   pip install -r requirements-databricks.txt
   ```

### 4. Configuration

The system uses a configuration file (`config-databricks.yaml`) to store settings. You can either:

1. Use the provided `databricks_notebook.py` which will generate the configuration file automatically, or
2. Manually create the configuration file based on the template in `config-databricks.yaml`

### 5. Running the Application

1. Open the `databricks_notebook.py` file in your Databricks workspace.
2. Run the notebook cells sequentially to:
   - Install dependencies
   - Configure the system
   - Create the necessary catalog and schema
   - Initialize the system
   - Run queries against the system

## Usage

### Basic Query

```python
from app_databricks import process_query
import asyncio

query = "What are Google's carbon emissions goals?"
response = asyncio.run(process_query(query))
print(response)
```

### Interactive Interface

The `databricks_notebook.py` file includes an interactive interface using Databricks widgets that allows you to enter queries and see responses directly in the notebook.

## Customization

### Using Different Documents

To use different documents:

1. Upload your documents to the Databricks FileStore or DBFS
2. Update the `file` path in the configuration to point to your document
3. Set `load_documents` to `True` to process the new document

### Using Different Models

To use different Azure OpenAI models:

1. Update the deployment names in the configuration file to match your Azure OpenAI deployments
2. Ensure the models have similar capabilities to the ones specified in the original project

## Troubleshooting

### Common Issues

1. **Vector Search Endpoint Creation Fails**
   - Ensure you have the necessary permissions to create Vector Search endpoints
   - Check that Unity Catalog is enabled in your workspace

2. **Document Processing Fails**
   - Verify that your Azure Document Intelligence endpoint and key are correct
   - Check that the document format is supported by Azure Document Intelligence

3. **LLM Calls Fail**
   - Verify that your Azure OpenAI endpoint is correct
   - Check that the specified deployments exist in your Azure OpenAI resource

## Contributing

Contributions to improve the Databricks implementation are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests. 
