from config.config import Config
from config.config_model import ConfigModel
from src_code.collect.collect_01 import (get_season_data, get_team_list, get_game_list, get_boxscore_list,
                                         get_playbyplay_data, get_player_names)
from src_code.curate.curate_01 import curate_data
from src_code.model.model_01 import model_data, model_visualization
from src_code.model.model_02 import gnn


def main():
    config_dict = {
        "verbose": False,
        "produce_csv": False,
        "season_count": 1,
        "delete_files": False,
        "reload_seasons": False,
        "reload_teams": False,
        "reload_games": False,
        "update_game_statuses": True,
        "reload_boxscores": False,
        "reload_players": False,
        "reload_playernames": False,
        "reload_playbyplay": False,
        "reload_rosters": False,
        "reload_curate": False,
    }
    config = Config(config_dict)
    config_model = ConfigModel()
    get_data(config)
    curate_data(config)
    model_data(config)
    # model_visualization(config)
    # model, metrics = gnn(config, config_model)


def get_data(config):
    get_season_data(config)
    get_team_list(config)
    get_game_list(config)
    get_boxscore_list(config)
    get_player_names(config)
    get_playbyplay_data(config)


if __name__ == '__main__':
    main()
