from src_code.utils.utils import period_time_to_game_time, game_time_to_period_time, create_player_dict, decompose_lines, create_roster_dicts, create_player_stats
import copy

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
    teams = []
    period_id = []
    time_index = []
    event_id = []
    shift_id = []
    time_on_ice = []
    teams_empty_net = []
    delayed_penalty = []
    skaters = []
    away_players_stats = []
    home_players_stats = []

    player_list, player_dict = create_player_dict(data_names)

    i_shift = 0
    event_categ = config.event_categ
    shift_categ = config.shift_categ
    for i_game, game in enumerate(data_plays):
        away_team = data_games[i_game]['awayTeam']
        home_team = data_games[i_game]['homeTeam']
        away_players, home_players = create_roster_dicts(data_game_roster[i_game], away_team, home_team)
        last_event = None
        last_event_details = None
        last_compare_shift = None
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
                last_event = copy.deepcopy(event)
                last_event_details = copy.deepcopy(event_details)
                last_compare_shift = copy.deepcopy(compare_shift)
            elif event_details['event_name'] == 'hit' and shift_details['shift_name'] == 'hit' and game_time_event == game_time_shift:
                last_game_time_event = period_time_to_game_time(last_event['period'], last_event['game_time'])
                game_id.append(game[i_event]['game_id'])
                game_date.append(data_games[i_game]['game_date'])
                teams.append((away_team, home_team))
                period_id.append(event['period'])
                time_index.append(event['game_time'])
                event_id.append(i_event)
                shift_id.append(i_shift)

                time_on_ice.append(game_time_shift - last_game_time_event)
                away_empty_net = True
                away_skaters = 0
                for away_player in last_compare_shift['away_players']:
                    if away_player[1] == 'G':
                        away_empty_net = False
                    else:
                        away_skaters += 1

                home_empty_net = True
                home_skaters = 0
                for home_player in last_compare_shift['home_players']:
                    if home_player[1] == 'G':
                        home_empty_net = False
                    else:
                        home_skaters += 1

                skaters.append((away_skaters, home_skaters))
                teams_empty_net.append((away_empty_net, home_empty_net))

                face_off_winner = None
                face_off_loser = None
                if last_event_details == 'faceoff':
                    face_off_winner = last_event['face_off_winner']
                    face_off_loser = last_event['face_off_loser']

                for away_player in last_compare_shift['away_players']:
                    away_player_stats = create_player_stats(away_players[int(away_player[0])]['player_id'])
                    if face_off_winner is not None:
                        if away_player_stats['player_id'] == face_off_winner or away_player_stats['player_id'] == face_off_loser:
                            away_player_stats['faceoff'] = 1
                            if away_player_stats['player_id'] == face_off_winner:
                                away_player_stats['face_off_winner'] = 1

                away_players_stats.append(away_player_stats)


                for home_player in last_compare_shift['home_players']:
                    home_player_stats = create_player_stats(away_players[int(home_player[0])]['player_id'])
                    if face_off_winner is not None:
                        if home_player_stats['player_id'] == face_off_winner or home_player_stats['player_id'] == face_off_loser:
                            home_player_stats['faceoff'] = 1
                            if home_player_stats['player_id'] == face_off_winner:
                                home_player_stats['face_off_winner'] = 1

                home_players_stats.append(home_player_stats)



                last_event = event
                last_compare_shift = compare_shift
            i_shift += 1



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


