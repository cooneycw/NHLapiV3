class Player:
    _players = dict()

    def __init__(self, player_id, last_name, first_name):
        self.player_id = player_id
        self.last_name = last_name
        self.first_name = first_name
        self.seasons = set()
        self.teams = set()
        self.games = set()

    @classmethod
    def create_player(cls, player_id, last_name, first_name):
        if player_id in cls._players:
            print(f"Player with ID {player_id}:{last_name}:{first_name} already exists.")
            # tested..functions correctly.
            return None
        new_player = cls(player_id, last_name, first_name)
        cls._players[player_id] = new_player
        return new_player

    @classmethod
    def get_players(cls):
        return cls._players
