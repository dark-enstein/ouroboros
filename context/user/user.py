from core.db import get_user
from core.type import UserData
from core.models import Engine

class User:
    tree = UserData
    engine = Engine
    def embedding(self):
        # concat all other individual gets
        tree = get_user()

    def get_profile_embedding(self):
        return self.tree.profile
    def get_real_graph(self):
        return self.tree.socialgraph
    def get_anti_signals(self):
        return self.tree.anti_filters

    def filter(self, hydrated_posts):
        return self.engine.rank_k_profiles(hydrated_posts, self.tree.profile, self.tree.socialgraph, self.tree.anti_filters)

    def get_engagement(self):
        pass
    
    def update_filters(self, engagement):
        pass