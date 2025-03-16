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
        # Log unknown parameters to help debug config issues
        known_params = {"verbose", "produce_csv", "days_list", "seg_games", "season_count",
                        "delete_files", "reload_seasons", "reload_teams", "reload_games",
                        "update_game_statuses", "reload_boxscores", "reload_players",
                        "reload_playernames", "reload_playbyplay", "reload_rosters", "reload_curate"}

        unknown_params = set(input_dict.keys()) - known_params
        if unknown_params:
            print(f"Warning: Unknown configuration parameters: {unknown_params}")

        self.verbose = input_dict['verbose']
        self.produce_csv = input_dict['produce_csv']
        self.stat_window_sizes = [5, 10, 20, 40, 82]
        self.curr_date = datetime.now().date()
        self.split_data = datetime(2023, 7, 1).date()
        # self.curr_date = datetime(2024, 12, 1).date()
        self.max_workers = 28
        self.base_url = "https://api-web.nhle.com"
        self.base_url_lines = "https://www.dailyfaceoff.com"
        self.headers_lines = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"
        }
        self.endpoints = {
            "seasons": "{base_url}/v1/season",
            "standings": "{base_url}/v1/standings/now",
            "schedule": "{base_url}/v1/club-schedule-season/{team}/{season}",
            "boxscore_v1": "{base_url}/v1/gamecenter/{game_id}/boxscore",
            "boxscore_v2": "{base_url}/v1/gamecenter/{game_id}/right-rail",
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
            "all_game_rosters": os.path.join(self.current_path, "storage", "pickles", "all_game_rosters.pkl"),
            "all_curated": os.path.join(self.current_path, "storage", "pickles", "all_curated.pkl"),
            "all_curated_data": os.path.join(self.current_path, "storage", "pickles", "all_curated_data.pkl"),
            "game_output_csv": os.path.join(self.current_path, "storage", "output", "csv", "game_output"),
            "game_output_pkl": os.path.join(self.current_path, "storage", "output", "pkl", "game_output"),
            "game_output_jpg": os.path.join(self.current_path, "storage", "output", "jpg"),
            "gnn_analysis": os.path.join(self.current_path, "storage", "analysis", "gnn_analysis"),
            "summary_excel": os.path.join(self.current_path, "storage", "output", "excel", "summary_excel"),
            "graph": os.path.join(self.current_path, "storage", "output", "graph", "graph"),
        }

        self.days_list = input_dict.get("days_list", None)
        self.seg_games = input_dict.get("seg_games", None)
        self.season_count = input_dict.get("season_count", 3)
        self.delete_files = input_dict.get("delete_files", False)
        self.reload_seasons = input_dict.get("reload_seasons", False)
        self.reload_teams = input_dict.get("reload_teams", False)
        self.reload_games = input_dict.get("reload_games", False)
        self.update_game_statuses = input_dict.get("update_game_statuses", True)
        self.reload_boxscores = input_dict.get("reload_boxscores", False)
        self.reload_players = input_dict.get("reload_players", False)
        self.reload_playernames = input_dict.get("reload_playernames", False)
        self.reload_playbyplay = input_dict.get("reload_playbyplay", False)
        self.reload_rosters = input_dict.get("reload_rosters", False)
        self.reload_curate = input_dict.get("reload_curate", False)
        self.Season = Season
        self.Team = Team
        self.Game = Game
        self.Player = Player
        self.PlayerGameStats = PlayerGameStats
        self.PlayByPlay = PlayByPlay
        self.rosters = None
        self.player_list = None
        self.lines = None
        self.event_categ = self.event_registry()
        self.shift_categ = self.shift_registry()
        self.stat_attributes = {
            'team_stats': [
                'win', 'loss', 'faceoff_taken', 'faceoff_won', 'shot_attempt', 'shot_missed',
                'shot_blocked', 'shot_on_goal', 'shot_saved', 'shot_missed_shootout',
                'goal', 'goal_against', 'giveaways', 'takeaways', 'hit_another_player',
                'hit_by_player', 'penalties', 'penalties_served', 'penalties_drawn',
                'penalty_shot', 'penalty_shot_goal', 'penalty_shot_saved', 'penalties_duration'
            ],
            'player_stats': [
                'toi', 'faceoff_taken', 'faceoff_won', 'shot_attempt', 'shot_missed',
                'shot_blocked', 'shot_on_goal', 'shot_saved', 'shot_missed_shootout',
                'goal', 'assist', 'point', 'goal_against', 'giveaways', 'takeaways',
                'hit_another_player', 'hit_by_player', 'penalties', 'penalties_served',
                'penalties_drawn', 'penalty_shot', 'penalty_shot_goal', 'penalty_shot_saved',
                'penalties_duration'
            ],
            'player_pair_stats': [
                'toi', 'faceoff_taken', 'faceoff_won', 'shot_on_goal', 'shot_saved',
                'goal', 'assist', 'hit_another_player', 'hit_by_player', 'penalties_duration'
            ]
        }

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
            return None
        except Exception as e:
            print(f"Error loading data from {self.file_paths[dimension]}: {str(e)}")
            return None

    def save_data(self, dimension, data):
        try:
            # Sort the data by game_id before saving
            if dimension in ["all_boxscores", "all_plays", "all_shifts", "all_game_rosters"]:
                # Sort based on the data structure
                if dimension == "all_boxscores":
                    sorted_data = sorted(data, key=lambda x: x['id'])
                elif dimension == "all_plays":
                    sorted_data = sorted(data, key=lambda x: x[0]['game_id'] if (
                            x and len(x) > 0 and 'game_id' in x[0]) else 0)
                elif dimension == "all_shifts":
                    # For shifts, find the game_id in the first item that has it
                    def get_shift_game_id(shift_list):
                        if not shift_list or len(shift_list) == 0:
                            return 0
                        for shift in shift_list:
                            if 'game_id' in shift:
                                return shift['game_id']
                        return 0

                    sorted_data = sorted(data, key=get_shift_game_id)
                elif dimension == "all_game_rosters":
                    # For rosters, use the game_id from the first item if available
                    sorted_data = sorted(data, key=lambda x: x[0]['game_id'] if (
                            x and len(x) > 0 and 'game_id' in x[0]) else 0)
                else:
                    sorted_data = data
            elif dimension == "all_curated":
                # For the curated games set, convert to a sorted list
                sorted_data = sorted(list(data))
                print(f"Saving {len(sorted_data)} curated game IDs to {self.file_paths[dimension]}")
            elif dimension == "all_curated_data":
                # For the consolidated game data dictionary
                sorted_data = data
                print(f"Saving consolidated data for {len(sorted_data)} games to {self.file_paths[dimension]}")
            else:
                # For other data types that don't need sorting
                sorted_data = data

            with open(self.file_paths[dimension], 'wb') as file:
                pickle.dump(sorted_data, file)

            # Only print this generic message for dimensions not handled with specific messages above
            if dimension not in ["all_curated", "all_curated_data"]:
                print(f"Saved sorted data to {self.file_paths[dimension]}")
        except Exception as e:
            print(f"Error saving data to {self.file_paths[dimension]}: {str(e)}")

    @staticmethod
    def shift_registry():
        shift_categ = dict()
        shift_categ['PGSTR'] = {
            'shift_name': 'pregame-start',
            'sport_stat': False,
        }
        shift_categ['PGEND'] = {
            'shift_name': 'pregame-end',
            'sport_stat': False,
        }
        shift_categ['ANTHEM'] = {
            'shift_name': 'pregame-anthem',
            'sport_stat': False,
        }
        shift_categ['PSTR'] = {
            'shift_name': 'period-start',
            'sport_stat': False,
        }
        shift_categ['FAC'] = {
            'shift_name': 'faceoff',
            'sport_stat': True,
        }
        shift_categ['HIT'] = {
            'shift_name': 'hit',
            'sport_stat': True,
        }
        shift_categ['GIVE'] = {
            'shift_name': 'giveaway',
            'sport_stat': True,
        }
        shift_categ['GOAL'] = {
            'shift_name': 'goal',
            'sport_stat': True,
        }
        shift_categ['SHOT'] = {
            'shift_name': 'shot-on-goal',
            'sport_stat': True,
        }
        shift_categ['MISS'] = {
            'shift_name': 'missed-shot',
            'sport_stat': True,
        }
        shift_categ['BLOCK'] = {
            'shift_name': 'blocked-shot',
            'sport_stat': True,
        }
        shift_categ['PENL'] = {
            'shift_name': 'penalty',
            'sport_stat': True,
        }
        shift_categ['STOP'] = {
            'shift_name': 'stoppage',
            'sport_stat': True,
        }
        shift_categ['CHL'] = {
            'shift_name': 'stoppage',
            'sport_stat': False,
        }
        shift_categ['PEND'] = {
            'shift_name': 'period-end',
            'sport_stat': True,
        }
        shift_categ['EISTR'] = {
            'shift_name': 'game-end',
            'sport_stat': False,
        }
        shift_categ['EIEND'] = {
            'shift_name': 'game-end',
            'sport_stat': False,
        }
        shift_categ['SOC'] = {
            'shift_name': 'shootout-complete',
            'sport_stat': False,
        }
        shift_categ['TAKE'] = {
            'shift_name': 'takeaway',
            'sport_stat': True,
        }
        shift_categ['SPC'] = {
            'shift_name': 'unknown',
            'sport_stat': False,
        }
        shift_categ['GEND'] = {
            'shift_name': 'game-end',
            'sport_stat': False,
        }
        shift_categ['DELPEN'] = {
            'shift_name': 'delayed-penalty',
            'sport_stat': True,
        }
        return shift_categ

    @staticmethod
    def event_registry():
        event_categ = dict()
        event_categ[502] = {
            'event_name': 'faceoff',
            'sport_stat': True,
        }
        event_categ[503] = {
            'event_name': 'hit',
            'sport_stat': True,
        }
        event_categ[504] = {
            'event_name': 'giveaway',
            'sport_stat': True,
        }
        event_categ[505] = {
            'event_name': 'goal',
            'sport_stat': True,
        }
        event_categ[506] = {
            'event_name': 'shot-on-goal',
            'sport_stat': True,
        }
        event_categ[507] = {
            'event_name': 'missed-shot',
            'sport_stat': True,
        }
        event_categ[508] = {
            'event_name': 'blocked-shot',
            'sport_stat': True,
        }
        event_categ[509] = {
            'event_name': 'penalty',
            'sport_stat': True,
        }
        event_categ[510] = {
            'event_name': '',
            'sport_stat': True,
        }
        event_categ[516] = {
            'event_name': 'stoppage',
            'sport_stat': True,
        }
        event_categ[520] = {
            'event_name': 'game-start',
            'sport_stat': False,
        }
        event_categ[521] = {
            'event_name': 'period-end',
            'sport_stat': True,
        }
        event_categ[523] = {
            'event_name': 'shootout-complete',
            'sport_stat': False,
        }
        event_categ[524] = {
            'event_name': 'game-end',
            'sport_stat': False,
        }
        event_categ[525] = {
            'event_name': 'takeaway',
            'sport_stat': True,
        }
        event_categ[535] = {
            'event_name': 'delayed-penalty',
            'sport_stat': True,
        }
        event_categ[537] = {
            'event_name': 'penalty-shot-missed',
            'sport_stat': True,
        }
        return event_categ


