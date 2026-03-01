
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Text input cannot be empty")
        embedding = self.model.encode([text])[0]
        return embedding

    


def verify_model():
    search_instance = SemanticSearch()[0]
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
 
