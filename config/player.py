class Player:
    _players = dict()

    def __init__(self, player_id):
        self.player_id = player_id
        self.sweater_number = None
        self.last_name = None
        self.first_name = None
        self.seasons = set()
        self.teams = set()
        self.games = set()
        self.player_games = dict()

    def update_player(self, game_id, game_date, team, season, results):
        self.games.add(game_id)
        self.seasons.add(season)
        self.teams.add(team)
        playergame_obj = PlayerGameStats.create_playergame(self.player_id, game_id, game_date, team)
        self.player_games[game_id] = playergame_obj
        playergame_obj.update_playergame(results)

    def update_name(self, last_name, first_name):
        self.last_name = last_name
        self.first_name = first_name

    @classmethod
    def create_player(cls, player_id):
        if player_id in cls._players:
            # print(f"Player with ID {player_id} already exists.")
            # tested..functions correctly.
            return cls._players[player_id]
        new_player = cls(player_id)
        cls._players[player_id] = new_player
        return new_player

    @classmethod
    def get_players(cls):
        return cls._players

    @classmethod
    def get_player(cls, player_id):
        return cls._players[player_id]


class PlayerGameStats:
    _playergames = dict()

    def __init__(self, player_id, game_id, game_date, team):
        self.player_id = player_id
        self.game_id = game_id
        self.game_date = game_date
        self.team = team
        self.sweater_number = None
        self.goals = None
        self.assists = None
        self.points = None
        self.plus_minus = None
        self.pim = None
        self.hits = None
        self.ppg = None
        self.shots = None
        self.faceoff_pctg = None
        self.toi = None
        self.es_shots = None
        self.pp_shots = None
        self.sh_shots = None
        self.saves = None
        self.goals_against = None
        self.es_goals_against = None
        self.pp_goals_against = None
        self.sh_goals_against = None
        self.starter = None

    def update_playergame(self, results):
        self.sweater_number = results['sweater_number']
        self.goals = results.get('goals', 0)
        self.assists = results.get('assists', 0)
        self.points = results.get('points', 0)
        self.plus_minus = results.get('plus_minus', 0)
        self.pim = results.get('pim', 0)
        self.hits = results.get('hits', 0)
        self.ppg = results.get('ppg', 0)
        self.shots = results.get('shots',0)
        self.faceoff_pctg = results.get('faceoff_pctg', 0)
        self.toi = results.get('toi', 0)
        self.es_shots = results.get('es_shots', 0)
        self.pp_shots = results.get('pp_shots', 0)
        self.sh_shots = results.get('sh_shots', 0)
        self.saves = results.get('save_shots', 0)
        self.goals_against = results.get('goals_against', 0)
        self.es_goals_against = results.get('es_goals_against', 0)
        self.pp_goals_against = results.get('pp_goals_against', 0)
        self.sh_goals_against = results.get('sh_goals_against', 0)
        self.starter = results.get('starter', False)

    @classmethod
    def create_playergame(cls, player_id, game_id, game_date, team):
        player_game_date = (player_id, game_id, game_date)
        if player_game_date in cls._playergames:
            # print(f"Player with ID {player_id}:{game_id}:{game_date} already exists.")
            # tested..functions correctly.
            return cls._playergames[player_game_date]
        new_player_game = cls(player_id, game_id, game_date, team)
        cls._playergames[player_game_date] = new_player_game
        return new_player_game
