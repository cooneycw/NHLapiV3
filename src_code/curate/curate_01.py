from src_code.utils.utils import period_time_to_game_time, game_time_to_period_time, create_player_dict, create_roster_dicts, create_ordered_roster, create_player_stats
import copy
import pandas as pd

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

    player_list, player_dict = create_player_dict(data_names)

    event_categ = config.event_categ
    shift_categ = config.shift_categ
    for i_game, game in enumerate(data_plays):
        # if data_games[i_game]['id'] !=  2024020016:
        #     continue
        # else:
        #     cwc = 0
        i_shift = 0
        game_id = []
        game_date = []
        away_teams = []
        home_teams = []
        period_id = []
        time_index = []
        toi_list = []
        event_id = []
        shift_id = []
        away_empty_net = []
        home_empty_net = []
        away_skaters = []
        home_skaters = []
        player_data = []

        away_team = data_games[i_game]['awayTeam']
        home_team = data_games[i_game]['homeTeam']
        away_players, home_players = create_roster_dicts(data_game_roster[i_game], away_team, home_team)
        away_players_sorted, home_players_sorted = create_ordered_roster(data_game_roster[i_game], away_team, home_team)
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
                print('\n')
                print(f'i_shift: {i_shift}  i_event: {i_event}')
                print(f'event: {event["period"]} {event["elapsed_time"]} {event["event_type"]} {event["event_code"]} ')
                print(f'shift: {compare_shift["period"]} {compare_shift["elapsed_time"]} {shift_details["shift_name"]} {compare_shift["event_type"]} {shift_details["sport_stat"]}')
                print('\n')

                if shift_details is None:
                    cwc = 0
                if not shift_details['sport_stat']:
                    i_shift += 1
                    continue
                break

            game_time_shift = period_time_to_game_time(int(compare_shift['period']), compare_shift['game_time'])
            if (event_details['event_name'] == 'takeaway') and (shift_details['shift_name'] == 'giveaway') and (game_time_event == game_time_shift):
                cwc = 0
            if (event_details['event_name'] == 'giveaway') and (shift_details['shift_name'] == 'takeaway') and (game_time_event == game_time_shift):
                cwc = 0
            if (event_details['event_name'] == shift_details['shift_name']) and (game_time_event == game_time_shift):
                empty_net_data = process_empty_net(compare_shift)
                if event_details['event_name'] == 'faceoff':
                    toi, player_stats = process_faceoff(event, compare_shift, away_players, home_players)
                elif event_details['event_name'] == 'hit':  #503
                    toi, player_stats = process_hit(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'giveaway': #504
                    toi, player_stats = process_giveaway(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'takeaway': #504
                    toi, player_stats = process_takeaway(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'goal':  # 505
                    toi, player_stats = process_goal(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'shot-on-goal': #506
                    toi, player_stats = process_shot_on_goal(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'missed-shot':
                    toi, player_stats = process_missed_shot(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'blocked-shot':
                    toi, player_stats = process_blocked_shot(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'penalty':
                    toi, player_stats = process_penalty(event, compare_shift, away_players_sorted, home_players_sorted, last_event, game_time_event)
                elif event_details['event_name'] == 'stoppage':
                    toi, player_stats = process_stoppage(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'period-end':
                    toi, player_stats = process_period_end(event, compare_shift, away_players, home_players, last_event, game_time_event)


                #elif event_details['event_name'] == 'game-end':
                #    toi, player_stats = process_game_end(event, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'delayed-penalty':
                    toi, player_stats = process_delayed_penalty(event, compare_shift, away_players, home_players, last_event, game_time_event)
                else:
                    cwc = 0

                game_id.append(game[i_event]['game_id'])
                game_date.append(data_games[i_game]['game_date'])
                away_teams.append(away_team)
                home_teams.append(home_team)
                period_id.append(event['period'])
                time_index.append(event['game_time'])
                toi_list.append(toi)
                event_id.append(i_event)
                shift_id.append(i_shift)
                away_empty_net.append(empty_net_data['away_empty_net'])
                home_empty_net.append(empty_net_data['home_empty_net'])
                away_skaters.append(empty_net_data['away_skaters'])
                home_skaters.append(empty_net_data['home_skaters'])
                player_data.append(player_stats)
                last_event = copy.deepcopy(event)

                i_shift += 1

        data = {
            'game_id': game_id,
            'game_date': game_date,
            'away_teams': away_teams,
            'home_teams': home_teams,
            'period_id': period_id,
            'time_index': time_index,
            'time_on_ice': toi_list,
            'event_id': event_id,
            'shift_id': shift_id,
            'away_empty_net': away_empty_net,
            'home_empty_net': home_empty_net,
            'away_skaters': away_skaters,
            'home_skaters': home_skaters
        }

        # Step 4: Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data)

        player_attributes = list(player_data[0][0].keys())

        standardized_players = []

        # Sort away players by sweater number
        for player_id in sorted(away_players.keys()):
            standardized_players.append(away_players[player_id])

        # Sort home players by sweater number
        for player_id in sorted(home_players.keys()):
            standardized_players.append(home_players[player_id])

        num_players = len(standardized_players)
        # Initialize new columns with NaN for each player and attribute
        new_columns = {'player_data': player_data}

        # Generate dynamic key-value pairs and update the dictionary
        new_columns.update({
            f'player_{i}_{attr}': pd.NA
            for i in range(1, num_players + 1)
            for attr in player_attributes
        })

        # Add all new columns at once using .assign()
        new_columns_df = pd.DataFrame(new_columns, index=df.index)

        player_id_to_position = {}
        for idx, player in enumerate(standardized_players, start=1):
            player_id = player['player_id']
            player_id_to_position[player_id] = idx

        def populate_player_columns(row):
            # Assume 'player_list' is the column in df that contains the list of player dicts
            player_data_row = row['player_data']  # Adjust the column name as needed

            for player_in_row in player_data_row:
                player_id_for_player_in_row = player_in_row['player_id']
                if player_id_for_player_in_row in player_id_to_position:
                    pos = player_id_to_position[player_id_for_player_in_row]
                    for attr in player_attributes:
                        col_name = f'player_{pos}_{attr}'
                        row[col_name] = player_in_row.get(attr, pd.NA)
                else:
                    # Handle players not in the standardized list if necessary
                    pass
            return row

        new_columns_df = new_columns_df.apply(populate_player_columns, axis=1)

        new_columns_df = new_columns_df.drop(columns=['player_data'])

        # Step 2: Concatenate df and the modified new_columns_df along the columns
        df = pd.concat([df, new_columns_df], axis=1)

        attributes_to_sum = ['goal', 'assist', 'shot_on_goal', 'shot_on_goal_overtime',  'goal_against', 'goal_overtime', 'goal_shootout', 'penalties_duration', 'penalties_duration_overtime', 'hit_another_player', 'hit_by_player', 'giveaways', 'takeaways']  # Example attributes

        # Initialize columns for summed attributes

        away_columns = {attr: [] for attr in attributes_to_sum}
        home_columns = {attr: [] for attr in attributes_to_sum}

        # Populate the column lists
        for attr in attributes_to_sum:
            # Away players: player_1_attr to player_20_attr
            away_columns[attr] = [f'player_{i}_{attr}' for i in range(1, len(away_players) + 1)]

            # Home players: player_21_attr to player_40_attr
            home_columns[attr] = [f'player_{i}_{attr}' for i in range(len(away_players) + 1, 2 * len(away_players) + 1)]

        team_sums = {'away': {}, 'home': {}}

        # Calculate sums for each attribute
        for attr in attributes_to_sum:
            # Sum for away team
            sum_series = df[away_columns[attr]].sum().to_list()
            sum_list = [int(x) for x in sum_series]
            team_sums['away'][f'away_{attr}_sum'] = sum(sum_list)

            # Sum for home team
            sum_series = df[home_columns[attr]].sum().to_list()
            sum_list = [int(x) for x in sum_series]
            team_sums['home'][f'home_{attr}_sum'] = sum(sum_list)

        print(f'game_id: {game_id[0]}  toi: {df["time_on_ice"].sum()} {data_games[i_game]["playbyplay"]}')
        all_good = True
        reason = ''
        if data_games[i_game]['away_goals'] != (team_sums['away']['away_goal_sum'] + team_sums['away']['away_goal_overtime_sum']):
            all_good = False
            reason = 'away_goals'
        if data_games[i_game]['home_goals'] != (team_sums['home']['home_goal_sum'] + team_sums['home']['home_goal_overtime_sum']):
            all_good = False
            reason = 'home_goals'
        if data_games[i_game]['away_sog'] != (team_sums['away']['away_shot_on_goal_sum'] + team_sums['away']['away_shot_on_goal_overtime_sum']):
            all_good = False
            reason = 'away_sog'
        if data_games[i_game]['home_sog'] != (team_sums['home']['home_shot_on_goal_sum'] + team_sums['home']['home_shot_on_goal_overtime_sum']):
            all_good = False
            reason = 'home_sog'
        if data_games[i_game]['away_pim'] != (team_sums['away']['away_penalties_duration_sum'] + team_sums['away']['away_penalties_duration_overtime_sum']):
            all_good = False
            reason = 'away_pim'
        if data_games[i_game]['home_pim'] != (team_sums['home']['home_penalties_duration_sum'] + team_sums['home']['home_penalties_duration_overtime_sum']):
            all_good = False
            reason = 'home_pim'
        if data_games[i_game]['away_hits'] != team_sums['away']['away_hit_another_player_sum']:
            all_good = False
            reason = 'away_hits'
        if data_games[i_game]['home_hits'] != team_sums['home']['home_hit_another_player_sum']:
            all_good = False
            reason = 'home_hits'
        if data_games[i_game]['away_hits'] != team_sums['home']['home_hit_by_player_sum']:
            all_good = False
            reason = 'away_hits_v2'
        if data_games[i_game]['home_hits'] != team_sums['away']['away_hit_by_player_sum']:
            all_good = False
            reason = 'home_hits_v2'
        if data_games[i_game]['away_give'] != team_sums['away']['away_giveaways_sum']:
            all_good = False
            reason = 'away_giveaways'
        if data_games[i_game]['home_give'] != team_sums['home']['home_giveaways_sum']:
            all_good = False
            reason = 'home_giveaways'
        if data_games[i_game]['away_take'] != team_sums['away']['away_takeaways_sum']:
            all_good = False
            reason = 'away_giveaways'
        if data_games[i_game]['home_take'] != team_sums['home']['home_takeaways_sum']:
            all_good = False
            reason = 'home_giveaways'


        if not all_good:
            print(f'reason: {reason}')
            print(f'shift data: {team_sums["away"]}  {team_sums["home"]}')
            print(f'boxscore data: away_goals {data_games[i_game]["away_goals"]} away_sog {data_games[i_game]["away_sog"]}  home_goals {data_games[i_game]["home_goals"]} home_sog {data_games[i_game]["home_sog"]}')
        else:
            print(f'game totals confirmed')

        print('\n')
        # Step 4: Export the DataFrame to CSV
        df.to_csv(config.file_paths['game_output'] + f'{str(game_id[0])}.csv', na_rep='', index=False)

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
    toi = 0
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

    return toi, player_stats


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

    return toi, player_stats


def process_takeaway(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['takeaway'] == player_id['player_id']:
            away_player_stats['takeaways'] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['takeaway'] == player_id['player_id']:
            home_player_stats['takeaways'] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


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
            home_player_stats['giveaways'] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_goal(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    shot_attempt_code = ['shot_attempt', 'shot_attempt_overtime', 'shot_attempt_shootout'][period_code]
    shot_on_goal_code = ['shot_on_goal', 'shot_on_goal_overtime', 'shot_on_goal_shootout'][period_code]
    goal_code = ['goal', 'goal_overtime', 'goal_shootout'][period_code]
    assist_code = ['assist', 'assist_overtime', 'assist_shootout'][period_code]
    goal_against_code = ['goal_against', 'goal_against_overtime', 'goal_against_shootout'][period_code]

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['shot_attempt'] == player_id['player_id']:
            away_player_stats[shot_attempt_code] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            away_player_stats[shot_on_goal_code] += 1
        if event['goal'] == player_id['player_id']:
            away_player_stats[goal_code] += 1
        if event['goal_assist1'] == player_id['player_id']:
            away_player_stats[assist_code] += 1
        if event['goal_assist2'] == player_id['player_id']:
            away_player_stats[assist_code] += 1
        if event['goal_against'] == player_id['player_id']:
            away_player_stats[goal_against_code] += 1

        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['shot_attempt'] == player_id['player_id']:
            home_player_stats[shot_attempt_code] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            home_player_stats[shot_on_goal_code] += 1
        if event['goal'] == player_id['player_id']:
            home_player_stats[goal_code] += 1
        if event['goal_assist1'] == player_id['player_id']:
            home_player_stats[assist_code] += 1
        if event['goal_assist2'] == player_id['player_id']:
            home_player_stats[assist_code] += 1
        if event['goal_against'] == player_id['player_id']:
            home_player_stats[goal_against_code] += 1

        player_stats.append(home_player_stats)

    return toi, player_stats


def process_shot_on_goal(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    shot_attempt_code = ['shot_attempt', 'shot_attempt_overtime', 'shot_attempt_shootout'][period_code]
    shot_on_goal_code = ['shot_on_goal', 'shot_on_goal_overtime', 'shot_on_goal_shootout'][period_code]
    shot_saved_code = ['shot_saved', 'shot_saved_overtime', 'shot_saved_shootout'][period_code]

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['shot_attempt'] == player_id['player_id']:
            away_player_stats[shot_attempt_code] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            away_player_stats[shot_on_goal_code] += 1
        if event['shot_saved'] == player_id['player_id']:
            away_player_stats[shot_saved_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['shot_attempt'] == player_id['player_id']:
            home_player_stats[shot_attempt_code] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            home_player_stats[shot_on_goal_code] += 1
        if event['shot_saved'] == player_id['player_id']:
            home_player_stats[shot_saved_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_missed_shot(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    shot_missed_code = ['shot_missed', 'shot_missed_overtime', 'shot_missed_shootout'][period_code]

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['missed_shot_attempt'] == player_id['player_id']:
            away_player_stats[shot_missed_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['missed_shot_attempt'] == player_id['player_id']:
            home_player_stats[shot_missed_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_blocked_shot(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    shot_attempt_code = ['shot_attempt', 'shot_attempt_overtime', 'shot_attempt_shootout'][period_code]
    shot_blocked_code = ['shot_blocked', 'shot_blocked_overtime', 'shot_blocked_shootout'][period_code]

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['blocked_shot_attempt'] == player_id['player_id']:
            away_player_stats[shot_attempt_code] += 1
        if event['blocked_shot_saved'] == player_id['player_id']:
            away_player_stats[shot_blocked_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['blocked_shot_attempt'] == player_id['player_id']:
            home_player_stats[shot_attempt_code] += 1
        if event['blocked_shot_saved'] == player_id['player_id']:
            home_player_stats[shot_blocked_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_penalty(event, compare_shift, away_players_sorted, home_players_sorted, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    penalties_code = ['penalties', 'penalties_overtime', 'penalties_shootout'][period_code]
    penalties_served_code = ['penalties_served', 'penalties_served_overtime', 'penalties_served_shootout'][period_code]
    penalties_drawn_code = ['penalties_drawn', 'penalties_drawn_overtime', 'penalties_drawn_shootout'][period_code]
    penalties_duration_code = ['penalties_duration', 'penalties_duration_overtime', 'penalties_duration_shootout'][period_code]

    on_ice = [int(sweater[0]) for sweater in compare_shift['away_players']]
    for player_id in away_players_sorted.keys():
        player_data = away_players_sorted[player_id]
        away_player_stats = create_player_stats(player_data)
        if away_players_sorted[player_id]['player_sweater'] in on_ice:
            away_player_stats['toi'] += toi
        if event['penalty_served'] == player_id:
            away_player_stats[penalties_served_code] += 1
        if event['penalty_committed'] == player_id:
            away_player_stats[penalties_code] += 1
        if (event['penalty_committed'] == player_id) or (event['penalty_served'] == player_id):
            away_player_stats[penalties_duration_code] += event['penalty_duration']
        if event['penalty_drawn'] == player_id:
            away_player_stats[penalties_drawn_code] += 1

        player_stats.append(away_player_stats)

    on_ice = [int(sweater[0]) for sweater in compare_shift['home_players']]
    for player_id in home_players_sorted.keys():
        player_data = home_players_sorted[player_id]
        home_player_stats = create_player_stats(player_data)
        if home_players_sorted[player_id]['player_sweater'] in on_ice:
            home_player_stats['toi'] += toi
        if event['penalty_served'] == player_id['player_id']:
            home_player_stats[penalties_served_code] += 1
        if event['penalty_committed'] == player_id:
            home_player_stats[penalties_code] += 1
        if (event['penalty_committed'] == player_id) or (event['penalty_served'] == player_id):
            home_player_stats[penalties_duration_code] += event['penalty_duration']
        if event['penalty_drawn'] == player_id:
            home_player_stats[penalties_drawn_code] += 1

        player_stats.append(home_player_stats)

    return toi, player_stats


def process_miss(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    shot_attempt_code = ['shot_attempt', 'shot_attempt_overtime', 'shot_attempt_shootout'][period_code]
    shot_missed_code = ['shot_missed', 'shot_missed_overtime', 'shot_missed_shootout'][period_code]

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['miss'] == player_id['player_id']:
            away_player_stats[shot_attempt_code] += 1
        if event['miss'] == player_id['player_id']:
            away_player_stats[shot_missed_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['miss'] == player_id['player_id']:
            home_player_stats[shot_attempt_code] += 1
        if event['miss'] == player_id['player_id']:
            home_player_stats[shot_missed_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_stoppage(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)
    period_code = 0
    if event['overtime']:
        period_code = 1
    elif event['shootout']:
        period_code = 2
    icing_code = ['icing', 'icing_overtime', 'icing_shootout'][period_code]
    offside_code = ['offside', 'offside_overtime', 'offside_shootout'][period_code]


    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        if event['stoppage'] in ['icing', 'goalie-stopped-after-sog', 'puck-in-crowd', 'puck-in-netting',
                                 'offside', 'puck-in-crowd', 'puck-in-benches', 'puck-frozen', 'tv-timeout',
                                 'high-stick', 'net-dislodged-defensive-skater', 'player-injury', 'video-review',
                                 'referee-or-linesman', 'chlg-league-goal-interference',
                                 'hand-pass', 'objects-on-ice', 'goalie-puck-frozen-played-from-beyond-center',
                                 'visitor-timeout', 'net-dislodged-offensive-skater', 'chlg-hm-goal-interference',
                                 'chlg-vis-goal-interference', 'chlg-hm-missed-stoppage', 'chlg-vis-missed-stoppage',
                                 'skater-puck-frozen', 'ice-scrape', 'chlg-league-missed-stoppage',
                                 'player-equipment', 'chlg-hm-off-side', 'chlg-vis-off-side',
                                 'chlg-hm-missed-stoppage', 'home-timeout', 'clock-problem',
                                 'puck-in-penalty-benches', 'ice-problem', 'net-dislodged-by-goaltender',
                                 'rink-repair', 'official-injury']:
            pass  # data lacks detail to specify which goalie / team
        else:
            print(f'away stoppage reason: {event["stoppage"]}')
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        if event['stoppage'] in ['icing', 'goalie-stopped-after-sog', 'puck-in-crowd', 'puck-in-netting',
                                 'offside', 'puck-in-crowd', 'puck-in-benches', 'puck-frozen', 'tv-timeout',
                                 'high-stick', 'net-dislodged-defensive-skater', 'player-injury', 'video-review',
                                 'referee-or-linesman', 'chlg-league-goal-interference',
                                 'hand-pass', 'objects-on-ice', 'goalie-puck-frozen-played-from-beyond-center',
                                 'visitor-timeout', 'net-dislodged-offensive-skater', 'chlg-hm-goal-interference',
                                 'chlg-vis-goal-interference', 'chlg-hm-missed-stoppage', 'chlg-vis-missed-stoppage',
                                 'skater-puck-frozen', 'ice-scrape', 'chlg-league-missed-stoppage',
                                 'player-equipment', 'chlg-hm-off-side', 'chlg-vis-off-side',
                                 'chlg-hm-missed-stoppage', 'home-timeout', 'clock-problem',
                                 'puck-in-penalty-benches', 'ice-problem', 'net-dislodged-by-goaltender',
                                 'rink-repair', 'official-injury']:
            pass  # data lacks detail to specify which goalie / team
        else:
             print(f'home stoppage reason: {event["stoppage"]}')
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_period_end(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_game_end(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_delayed_penalty(event, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'] += toi
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'] += toi
        player_stats.append(home_player_stats)

    return toi, player_stats


def calc_toi(game_time_event, last_event):
    last_game_time_event = period_time_to_game_time(last_event['period'], last_event['game_time'])
    toi = game_time_event - last_game_time_event
    return toi
