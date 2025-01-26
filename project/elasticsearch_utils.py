from elasticsearch import Elasticsearch
from llama_index.core import Document

def connect_to_elasticsearch(url, username, password):
    """
    Kết nối đến Elasticsearch.
    """
    return Elasticsearch(
        url,
        http_auth=(username, password)
    )

def fetch_documents(es_client, index_name):
    """
    Lấy tài liệu từ Elasticsearch index.
    """
    response = es_client.search(index=index_name, body={"query": {"match_all": {}}})
    documents = [
        Document(
            text=hit["_source"]["content"],
            metadata=hit["_source"].get("metadata", {})
        )
        for hit in response["hits"]["hits"]
    ]
    return documents
