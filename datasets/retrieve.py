import json
from core.db import setup_db, Database
from core.type import Post
from core.models import Engine

class Retriever:
    def get():
        pass

class Embedder:
    def embed(self, raw_json):
        pass

class Embedding(Retriever):
    embedding = list[Post]
    engine = Engine
    post_classification = list[Post]
    def __init__(vector_db):
        embedding = get_embedding(vector_db)

    def get(self):
        return self.embedding
    
    def get_innetwork_posts(self, profile_embed):
        # some logistic regression, or clustering means or ranking algo
        return Engine.classify_posts(self.embedding, profile_embed)[0]

    def get_outnetwork_posts(self, profile_embed):
        # some logistic regression, or clustering means or ranking algo
        return Engine.classify_posts(self.embedding, profile_embed)[1]


def get_embedding(db):
    db = setup_db()

    return db.retrieve_all()