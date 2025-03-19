import gc
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

    def save_curated_data_seasons(self, dimension_base, data, season):
        """Save curated data separately for each season.

        Args:
            dimension_base: The base dimension name (e.g., "all_curated" or "all_curated_data")
            data: The data to save
            season: The season identifier to append to the filename
        """
        season_str = str(season)
        file_path = self.file_paths[dimension_base].replace(".pkl", f"_{season_str}.pkl")

        try:
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
            print(f"Saved {dimension_base} data for season {season_str} to {file_path}")
        except Exception as e:
            print(f"Error saving {dimension_base} data for season {season_str}: {str(e)}")

    def load_curated_data(self):
        """Load curated data from all season-specific files."""
        print(f"Loading curated data with season_count = {self.season_count}...")

        # First try using the Season class
        selected_seasons_result = self.Season.get_selected_seasons(self.season_count)

        # If that doesn't work, load seasons directly from disk
        if not selected_seasons_result:
            print("Season.get_selected_seasons returned empty list. Loading seasons from disk...")

            try:
                seasons_file = self.file_paths["all_seasons"]
                if os.path.exists(seasons_file):
                    with open(seasons_file, 'rb') as file:
                        all_seasons = pickle.load(file)

                    print(f"Loaded {len(all_seasons)} seasons from disk")

                    # Handle different possible season data structures
                    if isinstance(all_seasons, list):
                        # If the seasons are just integers
                        if all(isinstance(s, int) for s in all_seasons):
                            # Sort in descending order (newest first)
                            selected_seasons = sorted(all_seasons, reverse=True)[:self.season_count]
                        # If seasons are objects or dictionaries
                        else:
                            # Try multiple ways to extract and sort seasons
                            try:
                                # If seasons have an 'id' attribute
                                sorted_seasons = sorted(all_seasons,
                                                        key=lambda s: getattr(s, 'id', 0) if hasattr(s, 'id') else 0,
                                                        reverse=True)
                                selected_seasons = []
                                for s in sorted_seasons[:self.season_count]:
                                    if hasattr(s, 'id'):
                                        selected_seasons.append(s.id)
                                    else:
                                        # If no id, try to use the season object itself
                                        selected_seasons.append(s)
                            except:
                                # Fallback: just use the first N seasons
                                selected_seasons = all_seasons[:self.season_count]
                    else:
                        # If all_seasons is not a list, use hardcoded values
                        print(f"Unexpected seasons data format: {type(all_seasons)}")
                        selected_seasons = [2023, 2022, 2021][:self.season_count]
                else:
                    print(f"Seasons file not found: {seasons_file}")
                    selected_seasons = [2023, 2022, 2021][:self.season_count]
            except Exception as e:
                print(f"Error loading seasons from disk: {str(e)}")
                selected_seasons = [2023, 2022, 2021][:self.season_count]
        else:
            # Extract season IDs from the result
            selected_seasons = []
            for item in selected_seasons_result:
                if isinstance(item, tuple) and len(item) > 0:
                    selected_seasons.append(item[0])
                elif isinstance(item, (int, str)):
                    selected_seasons.append(item)
                elif hasattr(item, 'id'):
                    selected_seasons.append(item.id)
                else:
                    print(f"Skipping unknown season format: {item}")

        print(f"Using seasons: {selected_seasons}")

        # Initialize result containers
        all_game_ids = set()
        all_game_data = {}

        # First, try to load from the base "all_curated" and "all_curated_data" files
        print("Checking for base curated files...")
        base_files_found = False

        # Try base curated IDs file
        if os.path.exists(self.file_paths["all_curated"]):
            try:
                with open(self.file_paths["all_curated"], 'rb') as file:
                    loaded_ids = pickle.load(file)
                if isinstance(loaded_ids, list):
                    loaded_ids = set(loaded_ids)
                all_game_ids.update(loaded_ids)
                print(f"Loaded {len(loaded_ids)} curated game IDs from base file")
                base_files_found = True
            except Exception as e:
                print(f"Error loading curated IDs from base file: {str(e)}")

        # Try base curated data file
        if os.path.exists(self.file_paths["all_curated_data"]):
            try:
                with open(self.file_paths["all_curated_data"], 'rb') as file:
                    loaded_data = pickle.load(file)
                all_game_data.update(loaded_data)
                print(f"Loaded {len(loaded_data)} curated game data entries from base file")
                base_files_found = True
            except Exception as e:
                print(f"Error loading curated data from base file: {str(e)}")

        # If we found base files, we can skip season-specific files
        if base_files_found:
            print("Successfully loaded data from base files.")
        else:
            # Otherwise, try to load from season-specific files
            print("Trying season-specific files...")

            # Look for curated data files in the storage directory
            storage_dir = os.path.dirname(self.file_paths["all_curated"])
            print(f"Checking storage directory: {storage_dir}")

            if os.path.exists(storage_dir):
                # List all files in the directory
                all_files = os.listdir(storage_dir)

                # Look for curated_*.pkl and curated_data_*.pkl files
                curated_id_files = [f for f in all_files if f.startswith("all_curated_") and not f.startswith(
                    "all_curated_data_") and f.endswith(".pkl")]
                curated_data_files = [f for f in all_files if f.startswith("all_curated_data_") and f.endswith(".pkl")]

                print(
                    f"Found {len(curated_id_files)} curated ID files and {len(curated_data_files)} curated data files")

                # Process each curated ID file
                for file_name in curated_id_files:
                    try:
                        file_path = os.path.join(storage_dir, file_name)
                        with open(file_path, 'rb') as file:
                            loaded_ids = pickle.load(file)
                        if isinstance(loaded_ids, list):
                            loaded_ids = set(loaded_ids)
                        all_game_ids.update(loaded_ids)
                        print(f"Loaded {len(loaded_ids)} curated game IDs from {file_name}")
                    except Exception as e:
                        print(f"Error loading curated IDs from {file_name}: {str(e)}")

                # Process each curated data file
                for file_name in curated_data_files:
                    try:
                        file_path = os.path.join(storage_dir, file_name)
                        with open(file_path, 'rb') as file:
                            loaded_data = pickle.load(file)
                        all_game_data.update(loaded_data)
                        print(f"Loaded {len(loaded_data)} curated game data entries from {file_name}")
                    except Exception as e:
                        print(f"Error loading curated data from {file_name}: {str(e)}")
            else:
                print(f"Storage directory not found: {storage_dir}")

        print(f"Total curated game IDs: {len(all_game_ids)}")
        print(f"Total curated game data entries: {len(all_game_data)}")

        # Print the first few game IDs to verify we've loaded something
        if all_game_ids:
            sample_ids = list(all_game_ids)[:5]
            print(f"Sample game IDs: {sample_ids}")

        return all_game_ids, all_game_data

    def load_curated_data_for_season(self, season):
        """Load curated data for a specific season.

        Args:
            season: The season identifier (e.g., 2023)

        Returns:
            A tuple of:
            - set of curated game IDs for the season
            - dictionary of curated game data for the season
        """
        season_str = str(season)
        game_ids = set()
        game_data = {}

        # Load game IDs
        curated_path = self.file_paths["all_curated"].replace(".pkl", f"_{season_str}.pkl")
        if os.path.exists(curated_path):
            try:
                with open(curated_path, 'rb') as file:
                    season_ids = pickle.load(file)
                if isinstance(season_ids, list):
                    season_ids = set(season_ids)
                game_ids = season_ids
                print(f"Loaded {len(game_ids)} curated game IDs from season {season_str}")
            except Exception as e:
                print(f"Error loading curated IDs for season {season_str}: {str(e)}")

        # Load game data
        data_path = self.file_paths["all_curated_data"].replace(".pkl", f"_{season_str}.pkl")
        if os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as file:
                    season_data = pickle.load(file)
                game_data = season_data
                print(f"Loaded {len(game_data)} curated game data entries from season {season_str}")
            except Exception as e:
                print(f"Error loading curated data for season {season_str}: {str(e)}")

        return game_ids, game_data

    def load_single_record(self, dimension, index):
        """Load a single record from a dataset without loading the entire dataset in memory.

        Args:
            dimension: The dimension name (e.g., 'all_plays')
            index: The index of the record to load

        Returns:
            The record at the specified index, or None if not found
        """
        try:
            with open(self.file_paths[dimension], 'rb') as file:
                data = pickle.load(file)
                record = data[index] if 0 <= index < len(data) else None
                # Immediately clear the large data object
                del data
                return record
        except Exception as e:
            print(f"Error loading record from {dimension} at index {index}: {str(e)}")
            return None

    def load_game_play_data(self, game_id: int):
        """Load play data for a specific game ID."""
        try:
            with open(self.file_paths['all_plays'], 'rb') as file:
                all_plays = pickle.load(file)
                for plays in all_plays:
                    if plays and len(plays) > 0 and 'game_id' in plays[0] and plays[0]['game_id'] == game_id:
                        return plays
            return None
        except Exception as e:
            print(f"Error loading play data for game {game_id}: {str(e)}")
            return None
        finally:
            # Encourage garbage collection
            gc.collect()

    def load_game_boxscore_data(self, game_id: int):
        """Load boxscore data for a specific game ID."""
        try:
            with open(self.file_paths['all_boxscores'], 'rb') as file:
                all_games = pickle.load(file)
                for game in all_games:
                    if 'id' in game and game['id'] == game_id:
                        return game
            return None
        except Exception as e:
            print(f"Error loading boxscore data for game {game_id}: {str(e)}")
            return None
        finally:
            # Encourage garbage collection
            gc.collect()

    def load_game_shift_data(self, game_id: int):
        """Load shift data for a specific game ID."""
        try:
            with open(self.file_paths['all_shifts'], 'rb') as file:
                all_shifts = pickle.load(file)
                for shifts in all_shifts:
                    if shifts and len(shifts) > 0 and 'game_id' in shifts[0] and shifts[0]['game_id'] == game_id:
                        return shifts
            return None
        except Exception as e:
            print(f"Error loading shift data for game {game_id}: {str(e)}")
            return None
        finally:
            # Encourage garbage collection
            gc.collect()

    def load_game_roster_data(self, game_id: int):
        """Load roster data for a specific game ID."""
        try:
            with open(self.file_paths['all_game_rosters'], 'rb') as file:
                all_rosters = pickle.load(file)
                for roster in all_rosters:
                    if roster and len(roster) > 0 and 'game_id' in roster[0] and roster[0]['game_id'] == game_id:
                        return roster
            return None
        except Exception as e:
            print(f"Error loading roster data for game {game_id}: {str(e)}")
            return None
        finally:
            # Encourage garbage collection
            gc.collect()

    def load_game_specific_data(self, game_id: int):
        """Load all data specific to a game ID."""
        play_data = self.load_game_play_data(game_id)
        game_data = self.load_game_boxscore_data(game_id)
        shift_data = self.load_game_shift_data(game_id)
        roster_data = self.load_game_roster_data(game_id)

        return play_data, game_data, shift_data, roster_data

    def identify_available_games(self):
        """Identify games that have data in all required datasets."""
        play_game_ids = set()
        boxscore_game_ids = set()
        shift_game_ids = set()
        roster_game_ids = set()

        # Get play game IDs
        try:
            data = self.load_data('all_plays')
            if data:
                for plays in data:
                    if plays and len(plays) > 0 and 'game_id' in plays[0]:
                        play_game_ids.add(plays[0]['game_id'])
        except Exception as e:
            print(f"Error loading play game IDs: {str(e)}")

        # Get boxscore game IDs
        try:
            data = self.load_data('all_boxscores')
            if data:
                for game in data:
                    if 'id' in game:
                        boxscore_game_ids.add(game['id'])
        except Exception as e:
            print(f"Error loading boxscore game IDs: {str(e)}")

        # Get shift game IDs
        try:
            data = self.load_data('all_shifts')
            if data:
                for shifts in data:
                    if shifts and len(shifts) > 0 and 'game_id' in shifts[0]:
                        shift_game_ids.add(shifts[0]['game_id'])
        except Exception as e:
            print(f"Error loading shift game IDs: {str(e)}")

        # Get roster game IDs
        try:
            data = self.load_data('all_game_rosters')
            if data:
                for roster in data:
                    if roster and len(roster) > 0 and 'game_id' in roster[0]:
                        roster_game_ids.add(roster[0]['game_id'])
        except Exception as e:
            print(f"Error loading roster game IDs: {str(e)}")

        # Force garbage collection
        gc.collect()

        # Return games that have data in all datasets
        return play_game_ids.intersection(boxscore_game_ids, shift_game_ids, roster_game_ids)

    def get_game_to_season_mapping(self):
        """Create mapping of game_id to season_id."""
        game_to_season = {}

        try:
            data = self.load_data('all_boxscores')
            if data:
                for game in data:
                    if 'id' in game:
                        game_id = game['id']
                        # Get the game object to access season_id
                        game_obj = self.Game.get_game(game_id)
                        if game_obj and hasattr(game_obj, 'season_id'):
                            game_to_season[game_id] = game_obj.season_id
        except Exception as e:
            print(f"Error creating game to season mapping: {str(e)}")

        # Force garbage collection
        gc.collect()

        return game_to_season

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


