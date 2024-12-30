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
    player_data = []

    player_list, player_dict = create_player_dict(data_names)

    i_shift = 0
    event_categ = config.event_categ
    shift_categ = config.shift_categ
    for i_game, game in enumerate(data_plays):
        away_team = data_games[i_game]['awayTeam']
        home_team = data_games[i_game]['homeTeam']
        away_players, home_players = create_roster_dicts(data_game_roster[i_game], away_team, home_team)
        last_event = None

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

            if (event_details['event_name'] == shift_details['shift_name']) and (game_time_event == game_time_shift):
                empty_net_data = process_empty_net(compare_shift)
                if event_details['event_name'] == 'faceoff':
                    player_stats = process_faceoff(event, compare_shift, away_players, home_players)
                elif event_details['event_name'] == 'hit':  #503
                    player_stats = process_hit(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'giveaway': #504
                    player_stats = process_giveaway(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'goal':  # 505
                    player_stats = process_giveaway(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'shot-on-goal': #506
                    player_stats = process_shot_on_goal(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'missed-shot':
                    player_stats = process_missed_shot(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'blocked-shot':
                    player_stats = process_blocked_shot(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'penalty':
                    player_stats = process_penalty(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'stop':
                    player_stats = process_stoppage(event, compare_shift, away_players, home_players, last_event, game_time_event)

                else:
                    cwc = 0

                game_id.append(game[i_event]['game_id'])
                game_date.append(data_games[i_game]['game_date'])
                teams.append((away_team, home_team))
                period_id.append(event['period'])
                time_index.append(event['game_time'])
                event_id.append(i_event)
                shift_id.append(i_shift)

                player_data.append(player_stats)
                last_event = copy.deepcopy(event)

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

def process_empty_net(compare_shift):
    ret_dict = {}

    away_empty_net = True
    away_skaters = 0
    for away_player in compare_shift['away_players']:
        if away_player[1] == 'G':
            away_empty_net = False
        else:
            away_skaters += 1

    home_empty_net = True
    home_skaters = 0
    for home_player in compare_shift['home_players']:
        if home_player[1] == 'G':
            home_empty_net = False
        else:
            home_skaters += 1

    ret_dict['away_empty_net'] = away_empty_net
    ret_dict['away_skaters'] = away_skaters
    ret_dict['home_empty_net'] = home_empty_net
    ret_dict['home_skaters'] = home_skaters

    return ret_dict


def process_faceoff(event, compare_shift, away_players, home_players):
    player_stats = []
    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        if (event['faceoff_winner'] == player_id['player_id']) or (event['faceoff_loser'] == player_id['player_id']):
            away_player_stats['faceoff_taken'] += 1
            if event['faceoff_winner'] == player_id['player_id']:
                away_player_stats['faceoff_won'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        if (event['faceoff_winner'] == player_id['player_id']) or (event['faceoff_loser'] == player_id['player_id']):
            home_player_stats['faceoff_taken'] += 1
            if event['faceoff_winner'] == player_id['player_id']:
                home_player_stats['faceoff_won'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_hit(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['hitting_player'] == player_id['player_id']:
            away_player_stats['hit_another_player'] += 1
        if event['hittee_player'] == player_id['player_id']:
            away_player_stats['hit_by_player'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['hitting_player'] == player_id['player_id']:
            home_player_stats['hit_another_player'] += 1
        if event['hittee_player'] == player_id['player_id']:
            home_player_stats['hit_by_player'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_giveaway(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['giveaway'] == player_id['player_id']:
            away_player_stats['giveaways'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['giveaway'] == player_id['player_id']:
            home_player_stats['giveaway'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_goal(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['giveaway'] == player_id['player_id']:
            away_player_stats['giveaways'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['giveaway'] == player_id['player_id']:
            home_player_stats['giveaway'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_shot_on_goal(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['shot_attempt'] == player_id['player_id']:
            away_player_stats['shot_attempt'] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            away_player_stats['shot_on_goal'] += 1
        if event['shot_saved'] == player_id['player_id']:
            away_player_stats['shot_saved'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['shot_attempt'] == player_id['player_id']:
            home_player_stats['shot_attempt'] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            home_player_stats['shot_on_goal'] += 1
        if event['shot_saved'] == player_id['player_id']:
            home_player_stats['shot_saved'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_missed_shot(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['missed_shot_attempt'] == player_id['player_id']:
            away_player_stats['shot_attempt'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['missed_shot_attempt'] == player_id['player_id']:
            home_player_stats['shot_attempt'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_blocked_shot(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['blocked_shot_attempt'] == player_id['player_id']:
            away_player_stats['shot_attempt'] += 1
        if event['blocked_shot_saved'] == player_id['player_id']:
            away_player_stats['shot_blocked'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['blocked_shot_attempt'] == player_id['player_id']:
            home_player_stats['shot_attempt'] += 1
        if event['blocked_shot_saved'] == player_id['player_id']:
            home_player_stats['shot_blocked'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_penalty(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['blocked_shot_attempt'] == player_id['player_id']:
            away_player_stats['shot_attempt'] += 1
        if event['blocked_shot_saved'] == player_id['player_id']:
            away_player_stats['shot_blocked'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['blocked_shot_attempt'] == player_id['player_id']:
            home_player_stats['shot_attempt'] += 1
        if event['blocked_shot_saved'] == player_id['player_id']:
            home_player_stats['shot_blocked'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_stoppage(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['shooting'] == player_id['player_id']:
            away_player_stats['hit_another_player'] += 1
        if event['hittee_player'] == player_id['player_id']:
            away_player_stats['hit_by_player'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['hitting_player'] == player_id['player_id']:
            home_player_stats['hit_another_player'] += 1
        if event['hittee_player'] == player_id['player_id']:
            home_player_stats['hit_by_player'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def process_miss(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['miss'] == player_id['player_id']:
            away_player_stats['hit_another_player'] += 1
        if event['hittee_player'] == player_id['player_id']:
            away_player_stats['hit_by_player'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['hitting_player'] == player_id['player_id']:
            home_player_stats['hit_another_player'] += 1
        if event['hittee_player'] == player_id['player_id']:
            home_player_stats['hit_by_player'] += 1
        player_stats.append(home_player_stats)

    return player_stats


def calc_toi(game_time_event, last_event):
    last_game_time_event = period_time_to_game_time(last_event['period'], last_event['game_time'])
    toi = game_time_event - last_game_time_event
    return toi
