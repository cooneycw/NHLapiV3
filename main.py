from config.config import Config
from src_code.collect.collect_01 import get_season_data, get_team_list, get_game_list, get_boxscore_list, get_playbyplay_data, get_player_names
from src_code.curate.curate_01 import curate_data

def main():
    config_dict = {
        "verbose": True,
        "season_count": 1,
        "delete_pickles": True,
        "reload_seasons": True,
        "reload_teams": True,
        "reload_games": True,
        "reload_boxscores": True,
        "reload_playernames": True,
        "reload_playbyplay": True,
        "reload_rosters": True,
    }
    config = Config(config_dict)
    get_data(config)
    curate_data(config)


def get_data(config):
    get_season_data(config)
    get_team_list(config)
    get_game_list(config)
    get_boxscore_list(config)
    get_player_names(config)
    get_playbyplay_data(config)


def curate_data_seg(curr_date, seg_list):
    curate_data(curr_date, seg_list)


if __name__ == '__main__':
    main()
