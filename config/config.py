import os
import pickle

from config.playbyplay import PlayByPlay
from config.season import Season
from config.team import Team
from config.game import Game
from config.player import Player, PlayerGameStats
from datetime import datetime


class Config:
    def __init__(self, input_dict):
        self.base_url = "https://api-web.nhle.com"
        self.base_url_lines = "https://www.dailyfaceoff.com"
        self.headers_lines = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"
        }
        self.endpoints = {
            "seasons": "{base_url}/v1/season",
            "standings": "{base_url}/v1/standings/now",
            "schedule": "{base_url}/v1/club-schedule-season/{team}/{season}",
            "boxscore": "{base_url}/v1/gamecenter/{game_id}/boxscore",
            "player": "{base_url}/v1/player/{player_id}/landing",
            "roster": "{base_url}/v1/roster/{team}/current",
            "lines": "{base_url_lines}/teams/{line_team}/line-combinations",
            "plays": "{base_url}/v1/gamecenter/{game_id}/play-by-play",
        }

        self.current_path = os.getcwd()
        self.file_paths = {
            "all_seasons": os.path.join(self.current_path, "storage", "pickles", "all_seasons.pkl"),
            "all_teams": os.path.join(self.current_path, "storage", "pickles", "all_teams.pkl"),
            "all_games": os.path.join(self.current_path, "storage", "pickles", "all_games.pkl"),
            "all_boxscores": os.path.join(self.current_path, "storage", "pickles", "all_boxscores.pkl"),
            "all_players": os.path.join(self.current_path, "storage", "pickles", "all_players.pkl"),
            "all_names": os.path.join(self.current_path, "storage", "pickles", "all_names.pkl"),
            "all_shifts": os.path.join(self.current_path, "storage", "pickles", "all_shifts.pkl"),
            "all_plays": os.path.join(self.current_path, "storage", "pickles", "all_plays.pkl"),
        }

        self.curr_date = datetime.now().date()
        self.days_list = input_dict.get("days_list", None)
        self.seg_games = input_dict.get("seg_games", None)
        self.season_count = input_dict.get("season_count", None)
        self.delete_files = input_dict.get("delete_files", None)
        self.reload_seasons = input_dict.get("reload_seasons", True)
        self.reload_teams = input_dict.get("reload_teams", True)
        self.reload_games = input_dict.get("reload_games", True)
        self.reload_boxscores = input_dict.get("reload_boxscores", True)
        self.reload_players = input_dict.get("reload_boxscores", True)
        self.reload_playernames = input_dict.get("reload_playernames", True)
        self.reload_playbyplay = input_dict.get("reload_playbyplay", True)
        self.Season = Season
        self.Team = Team
        self.Game = Game
        self.Player = Player
        self.PlayerGameStats = PlayerGameStats
        self.PlayByPlay = PlayByPlay
        self.rosters = None
        self.player_list = None
        self.lines = None

    def get_endpoint(self, key, **kwargs):
        """Construct and return the full URL for a given endpoint key with placeholders replaced by kwargs."""
        endpoint_template = self.endpoints[key]
        return endpoint_template.format(base_url=self.base_url, base_url_lines=self.base_url_lines, **kwargs)

    def del_data(self):
        for file_path in self.file_paths.values():
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found: {file_path}")

    def load_data(self, dimension):
        try:
            with open(self.file_paths[dimension], 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"Data not saved previously: {self.file_paths[dimension]}")
            return None, None
        except Exception as e:
            print(f"Error loading data from {self.file_paths[dimension]}: {str(e)}")
            return None, None

    def save_data(self, dimension, data):
        try:
            with open(self.file_paths[dimension], 'wb') as file:
                pickle.dump(data, file)
        except Exception as e:
            print(f"Error saving data to {self.file_paths[dimension]}: {str(e)}")

