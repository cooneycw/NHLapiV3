from src_code.utils.utils import period_time_to_game_time, game_time_to_period_time, create_player_dict


def curate_data(config):
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"
    dimension_games = "all_boxscores"
    dimension_players = "all_players"

    data_games = config.load_data(dimension_games)
    data_players = config.load_data(dimension_players)
    data_shifts = config.load_data(dimension_shifts)
    data_plays = config.load_data(dimension_plays)
    data_game_roster = config.load_data(dimension_game_rosters)

    game_id = []
    game_date = []
    home_id = []
    away_id = []
    period_id = []
    time_index = []
    shift_id = []
    time_on_ice = []
    player_dict = create_player_dict(data_players)

    i_shift = 0
    event_categ = config.event_categ
    for i_game, game in enumerate(data_plays):
        for i_event, event in enumerate(game):
            event_details = event_categ.get(event['event_code'])
            if event_details is None:
                cwc = 0
            if not event_details['sport_stat']:
                continue
            if event['away_players'] == [] and event['home_players'] == []:
                continue

            cwc = 0


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


