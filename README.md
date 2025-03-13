# Personal Chatbot

## Project Description

The Personal Chatbot project is an AI-driven conversational assistant designed to engage in natural, helpful conversations. It leverages advanced language models and vector databases to provide insightful responses and assist with various tasks. The chatbot is built using the Groq LLM, Pinecone for vector storage, and Hugging Face embeddings for semantic understanding.

## Features

- **Natural Language Processing**: Utilizes the Groq LLM for generating human-like responses.
- **Vector Database**: Integrates with Pinecone to store and retrieve document embeddings efficiently.
- **Embeddings**: Uses Hugging Face's sentence-transformers for creating embeddings.
- **Tool Integration**: Includes tools for file reading, writing, and SQL database querying.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd personal-chatbot
   ```

2. **Environment Variables**:
   - Create a `.env` file in the root directory.
   - Add your API keys:
     ```
     GROQ_API_KEY=your_groq_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     ```

3. **Install Dependencies**:
   - Ensure you have Python installed (preferably 3.8 or higher).
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Initialize Pinecone Index**:
   - The script will automatically create a Pinecone index if it doesn't exist. Ensure your Pinecone API key is set correctly.

## Usage

- **Start the Chatbot**:
  Run the chatbot using the following command:
  ```bash
  python digitalME.py
  ```
  Follow the on-screen instructions to interact with the chatbot.

- **Load Documents**:
  Uncomment the document loading section in `digitalME.py` to load documents into the vector store.

## File Structure

- `digitalME.py`: Main script for initializing and running the chatbot.
- `digitalME_GUI.py`: GUI version of the chatbot (if applicable).
- `transcript.db`: SQLite database for storing conversation transcripts.
- `personal/`: Directory for storing personal documents.
- `.env`: Environment variables file (not included in the repository).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Langchain](https://github.com/langchain-ai/langchain) for providing the language model framework.
- [Pinecone](https://www.pinecone.io/) for vector database services.
- [Hugging Face](https://huggingface.co/) for the embeddings model.
