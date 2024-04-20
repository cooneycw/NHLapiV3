class Season:
    _seasons = set()

    def __init__(self, season_id):
        self._seasons.add(self)
        self.season_id = season_id
        self.games = set()
        self.players = set()

    def add_game(self, game):
        self.games.add(game)

    def add_player(self, player):
        self.players.add(player)

    def get_games(self):
        return self.games

    @classmethod
    def get_seasons(cls):
        seasons = [(season.season_id, season) for season in list(cls._seasons)]
        sorted_tuples = sorted(seasons, key=lambda x: x[0])
        return sorted_tuples

    @classmethod
    def get_selected_seasons(cls, season_count):
        seasons = [(season.season_id, season) for season in list(cls._seasons)]
        sorted_tuples = sorted(seasons, key=lambda x: x[0])[-1 * season_count:]
        return sorted_tuples
