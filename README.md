# Herbal Remedies Assistant 🌿

Herbal Remedies Assistant is an interactive web application designed to provide users with informed answers about herbal remedies and natural medicine. The application leverages advanced natural language processing and vector search technologies to retrieve relevant information from a database of over 1000 herbal remedies.

## Features

- **Interactive Query Interface**: Ask questions about herbal remedies and receive detailed answers.
- **Vector Search**: Uses Chroma DB to find the most relevant information from the database.
- **Transparency**: View the source documents used to generate the answers.
- **Streamlit UI**: A user-friendly interface built with Streamlit.

## Technologies Used

- **[LangChain](https://www.langchain.com/)**: For building the retrieval chain and managing prompts.
- **[Chroma DB](https://www.trychroma.com/)**: For vector storage and search capabilities.
- **[Streamlit](https://streamlit.io/)**: For creating the web application interface.
- **[Ollama LLM](https://ollama.ai/)**: For generating natural language responses.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd HerbSphere
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your LangChain API key:
   ```env
   LANGCHAIN_API_KEY=your_api_key_here
   ```

4. Place the `herbal_remedies_1000plus.csv` file in the root directory.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your browser (Streamlit will provide a local URL).
2. Enter your question in the text input field (e.g., "What herbs help with anxiety?").
3. View the answer and the source documents for transparency.

## Project Structure

```
HerbSphere/
├── app.py                     # Main application file
├── herbal_remedies_1000plus.csv # Herbal remedies database (not included in the repository)
├── chroma_db/                 # Chroma DB vector storage
├── herbal_db/                 # Additional database files
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

## Database

Due to the large size of the database, it is not included in this repository. You can download the database from the following link:

**[Download Database](#)**

*(Replace `#` with the actual Google Drive link)*

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Special thanks to the creators of LangChain, Chroma DB, and Streamlit for their amazing tools.
- Made by Mudit Thakre.
