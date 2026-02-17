from __future__ import annotations
from core.db import setup_db, Database
from core.type import Post, ScoredPost, UserData
from core.models import Engine


class Retriever:
    def get(self) -> list[Post]:
        raise NotImplementedError


class Embedding(Retriever):
    def __init__(self):
        self.db = setup_db()
        self.in_posts: list[Post] = []
        self.out_posts: list[Post] = []
        self._load()

    def _load(self):
        self.in_posts, self.out_posts = self.db.retrieve_by_network()

    def get(self) -> list[Post]:
        return self.in_posts + self.out_posts

    def get_innetwork_posts(self) -> list[Post]:
        return self.in_posts

    def get_outnetwork_posts(self) -> list[Post]:
        return self.out_posts

    def classify(self, user: UserData) -> tuple[list[ScoredPost], list[ScoredPost]]:
        """Run full classification: embed + score + split by network."""
        return Engine.classify_posts(self.in_posts, self.out_posts, user)

    def hydrate(self, posts: list[Post]) -> list[Post]:
        """Hydrate lightweight post stubs with full content."""
        return [p.hydrate() for p in posts]
