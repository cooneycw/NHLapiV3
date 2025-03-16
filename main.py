from config.config import Config
from config.config_model import ConfigModel
from src_code.collect.collect_01 import (get_season_data, get_team_list, get_game_list, get_boxscore_list,
                                         get_playbyplay_data, get_player_names)
from src_code.curate.curate_01 import curate_data
from src_code.model.model_01 import model_data, model_visualization
from src_code.model.model_02 import run_gnn
from src_code.model.model_03 import run_gnn_enhanced


def main():
    config_dict = {
        "verbose": False,
        "produce_csv": False,
        "season_count": 3,
        "delete_files": True,
        "reload_seasons": True,
        "reload_teams": True,
        "reload_games": True,
        "update_game_statuses": True,
        "reload_boxscores": True,
        "reload_players": True,
        "reload_playernames": True,
        "reload_playbyplay": True,
        "reload_rosters": True,
        "reload_curate": True,
    }
    config = Config(config_dict)
    config_model = ConfigModel()
    get_data(config)
    curate_data(config)
    model_data(config)
    model_visualization(config)
    run_gnn_enhanced(config, config_model)


def get_data(config):
    get_season_data(config)
    get_team_list(config)
    get_game_list(config)
    get_boxscore_list(config)
    get_player_names(config)
    get_playbyplay_data(config)


if __name__ == '__main__':
    main()
