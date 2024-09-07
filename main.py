from config.config import Config
from src_code.collect.collect_01 import get_season_data, get_team_list, get_game_list, get_boxscore_list, get_playbyplay_data, get_playernames


def main():
    config_dict = {
        "season_count": 5,
        "delete_pickles": True,
        "reload_seasons": True,
        "reload_teams": True,
        "reload_games": True,
        "reload_boxscores": True,
        "reload_playernames": True,
        "reload_playbyplay": True,
    }
    config = Config(config_dict)
    get_data(config)


def get_data(config):
    get_season_data(config)
    get_team_list(config)
    get_game_list(config)
    get_boxscore_list(config)
    get_playernames(config)
    get_playbyplay_data(config)

    cwc = 0
    # asyncio.run(get_boxscore_list(config))
    # asyncio.run(get_rosters(config))
    # get_player_list(config)
    # save_data(config)


def curate_data_seg(curr_date, seg_list):
    curate_data(curr_date, seg_list)


def curate_data(curr_date, days_list):
    pass
    # config = load_data()
    # curate_basic_stats(config, curr_date)
    # curate_future_games(config, curr_date)
    # curate_player_stats(config, curr_date)
    # curate_future_player_stats(config, curr_date)
    # for days in days_list:
    #     print(f"Game processing days: {days}")
    #     curate_rolling_stats(config, curr_date, days=days)
    #     curate_proj_data(config, curr_date, days=days)
    #
    # first_days = True
    # for j, days in enumerate(days_list):
    #     if j != 0:
    #         first_days = False
    #     print(f"Player processing days: {days}")
    #     curate_rolling_player_stats(config, curr_date, first_days, days=days)
    #     curate_proj_player_data(config, curr_date, first_days, days=days)


if __name__ == '__main__':
    main()
