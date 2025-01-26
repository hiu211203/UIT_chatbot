# UIT Admissions Chatbot

Welcome to the UIT Admissions Chatbot project! This chatbot is designed to provide accurate and contextually relevant answers to queries related to admissions at the University of Information Technology (UIT). It uses OpenAI's GPT-3.5 model, Elasticsearch, and various natural language processing techniques for enhanced performance.

---

## Features

- **Multilingual Support**: Answers queries in Vietnamese or English based on the input language.
- **Context-Aware Responses**: Utilizes a retriever and reranker pipeline to ensure contextually accurate answers.
- **Custom Query Rewriting**: Rewrites user queries for improved clarity and precision.
- **History Summarization**: Summarizes past conversations to maintain concise and relevant interactions.
- **Streamlit Interface**: Simple and interactive user interface for engaging with the chatbot.

---

## Requirements

### Libraries and Tools

To run this project, you'll need the following Python libraries:

- `streamlit`
- `transformers`
- `llama-index`
- `elasticsearch`
- `openai`

## Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/uit-admissions-chatbot.git
   ```

2. Navigate to the project directory:

   ```bash
   cd uit-admissions-chatbot
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:

   Create a `.env` file in the root directory and add the following:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ELASTICSEARCH_URL=https://your_elasticsearch_url
   ELASTICSEARCH_USERNAME=your_username
   ELASTICSEARCH_PASSWORD=your_password
   ```

5. Start the Streamlit app:

   ```bash
   cd project
   streamlit run app.py
   ```

6. Open your browser at `http://localhost:8501` to interact with the chatbot.

---

## How It Works

1. **Query Input**: Users can input their questions through the Streamlit interface.
2. **Query Rewriting**: The chatbot rewrites the query for clarity.
3. **Document Retrieval**: Relevant documents are retrieved from Elasticsearch.
4. **Reranking**: Retrieved documents are reranked using a SentenceTransformer model.
5. **Response Generation**: GPT-3.5 generates the final response based on the best-matched context.
6. **History Management**: Past interactions are summarized and leveraged for better responses.

---

### Elasticsearch Index

Ensure your Elasticsearch instance is running, and the required index (`my_index`) is populated with documents containing fields like `content` and optional `metadata`.

---

## Future Improvements

- **Integrate More LLM Models**: Support for additional LLMs.
- **Enhanced UI/UX**: Improve user interface for better interactivity.
- **Extended Dataset**: Add more UIT-related documents to enhance response accuracy.
- **Performance Optimization**: Reduce latency for a smoother experience.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. Make sure to follow the coding standards and include relevant tests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- [OpenAI](https://openai.com/) for providing the GPT-3.5 model.
- [Streamlit](https://streamlit.io/) for the web app framework.
- [HuggingFace](https://huggingface.co/) for the transformers and embedding models.
- [Elasticsearch](https://www.elastic.co/) for document indexing and retrieval.

---

## Contact

For any questions or feedback, please reach out to [hieu211203@gmail.com](mailto:hieu211203@gmail.com).

---

Enjoy using the UIT Admissions Chatbot! 🚀

