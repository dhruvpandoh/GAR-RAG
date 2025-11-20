 # app/services/model_service.py
from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class ModelService:
    def __init__(self):
        # Initialize query enhancement model
        self.query_model = AutoModelForSeq2SeqGeneration.from_pretrained(
            settings.QUERY_ENHANCEMENT_MODEL
        )
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            settings.QUERY_ENHANCEMENT_MODEL
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    async def enhance_query(self, query: str) -> str:
        """Enhance the input query using the query enhancement model"""
        inputs = self.query_tokenizer(
            f"Enhance this search query: {query}",
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.query_model.generate(**inputs)
        enhanced_query = self.query_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return enhanced_query
    
    async def get_embeddings(self, text: str):
        """Get embeddings for the input text"""
        return self.embedding_model.encode(text)