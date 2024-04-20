class Game:
    _games = dict()

    def __init__(self, game_id, game_date, home_team, away_team, season_id):
        self.game_id = game_id
        self.game_date = game_date
        self.home_team = home_team
        self.away_team = away_team
        self.season_id = season_id
        self.play_by_play = None
        self.home_goals = None
        self.away_goals = None
        self.home_sog = None
        self.away_sog = None
        self.home_players = set()
        self.away_players = set()

    def update_game(self, input_dict):
        self.play_by_play = input_dict['play_by_play']
        self.home_goals = input_dict['home_goals']
        self.away_goals = input_dict['away_goals']
        self.home_sog = input_dict['home_sog']
        self.away_sog = input_dict['away_sog']

    def get_game_tuple(self):
        ret_tuple = (self.game_id, self.game_date, self.home_team, self.away_team, self.season_id)
        return ret_tuple

    @classmethod
    def create_game(cls, game_id, game_date, home_team, away_team, season_id):
        if game_id in cls._games:
            # print(f"Game with ID {game_id}:{home_team}:{away_team} already exists.")
            # tested - functions correctly.
            return None
        new_game = cls(game_id, game_date, home_team, away_team, season_id)
        cls._games[game_id] = new_game
        return new_game

    @classmethod
    def get_games(cls):
        games = [(game.game_id, game.game_date, game.home_team, game.away_team, game.season_id) for game in list(cls._games.values())]
        sorted_tuples = sorted(games, key=lambda x: x[0])
        return sorted_tuples

    @classmethod
    def get_game(cls, game_id):
        return cls._games[game_id]

