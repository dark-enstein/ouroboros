class Retriever:
    def get():
        pass

class Embedder:
    def embed(self, raw_csv):
        pass

class Embedding(Retriever):
    embedding = None
    def __init__(file):
        embedding = generate_embedding(file)

    def get(self):
        return self.embedding
    
    def get_innetwork_posts(self, profile_embed):
        # some logistic regression, or clustering means or ranking algo
        pass

    def get_outnetwork_posts(self, profile_embed):
        # some logistic regression, or clustering means or ranking algo
        pass


def generate_embedding(file):
    ember = Embedder()
    return ember.embed()