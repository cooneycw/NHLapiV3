from src_code.utils.utils import period_time_to_game_time, game_time_to_period_time, create_player_dict, decompose_lines, create_roster_dicts


def curate_data(config):
    dimension_names = "all_names"
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"
    dimension_games = "all_boxscores"
    dimension_players = "all_players"

    data_names = config.load_data(dimension_names)
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
    player_list, player_dict = create_player_dict(data_names)

    i_shift = 0
    event_categ = config.event_categ
    shift_categ = config.shift_categ
    for i_game, game in enumerate(data_plays):
        away_team = data_games[i_game]['awayTeam']
        home_team = data_games[i_game]['homeTeam']
        away_players, home_players = create_roster_dicts(data_game_roster[i_game], away_team, home_team)
        for i_event, event in enumerate(game):
            event_details = event_categ.get(event['event_code'])
            if event_details is None:
                cwc = 0
            if not event_details['sport_stat']:
                continue
            game_time_event = period_time_to_game_time(event['period'], event['game_time'])
            while True:
                compare_shift = data_shifts[i_game][i_shift]
                shift_details = shift_categ.get(compare_shift['event_type'])
                if shift_details is None:
                    cwc = 0
                if not shift_details['sport_stat']:
                    i_shift += 1
                    continue
                break
            game_time_shift = period_time_to_game_time(event['period'], event['game_time'])

            if event_details['event_name'] == 'faceoff' and shift_details['shift_name'] == 'faceoff' and game_time_event == game_time_shift:
                cwc = 0
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


