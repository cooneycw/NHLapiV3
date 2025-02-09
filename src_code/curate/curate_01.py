from functools import partial
from src_code.utils.utils import period_time_to_game_time, create_player_dict, create_roster_dicts, create_ordered_roster, create_player_stats, save_game_data
import concurrent.futures
import copy
import numpy as np
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
        # if data_games[i_game]['id'] !=  2022020158:
        #     continue
        # else:
        #     cwc = 0
        i_shift = 0
        game_id = []
        game_date = []
        away_teams = []
        home_teams = []
        period_id = []
        period_code = []
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

                if shift_details is None:
                    cwc = 0
                if not shift_details['sport_stat']:
                    i_shift += 1
                    continue
                break

            if config.verbose:
                print('\n')
                print(f'i_shift: {i_shift}  i_event: {i_event}')
                print(f'event: {event["period"]} {event["elapsed_time"]} {event["event_type"]} {event["event_code"]} ')
                print(f'shift: {compare_shift["period"]} {compare_shift["elapsed_time"]} {shift_details["shift_name"]} {compare_shift["event_type"]} {shift_details["sport_stat"]}')
                print('\n')

            game_time_shift = period_time_to_game_time(int(compare_shift['period']), compare_shift['game_time'])
            period_cd = 0
            if event['overtime']:
                period_cd = 1
            elif event['shootout']:
                period_cd = 2

            if (((event_details['event_name'] == shift_details['shift_name']) and (game_time_event == game_time_shift)) or
                ((event_details['event_name'] == 'penalty-shot') and (game_time_event == game_time_shift)) or
                ((event_details['event_name'] == 'penalty-shot-missed') and (game_time_event == game_time_shift))):
                empty_net_data = process_empty_net(compare_shift)
                if event_details['event_name'] == 'faceoff':
                    toi, player_stats = process_faceoff(event, period_cd, compare_shift, away_players, home_players)
                elif event_details['event_name'] == 'hit':  #503
                    toi, player_stats = process_hit(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'giveaway': #504
                    toi, player_stats = process_giveaway(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'takeaway': #504
                    toi, player_stats = process_takeaway(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'goal':  # 505
                    toi, player_stats = process_goal(event, period_cd, compare_shift, away_players, home_players, away_players_sorted, home_players_sorted, last_event, game_time_event)
                elif event_details['event_name'] == 'shot-on-goal': #506
                    toi, player_stats = process_shot_on_goal(config.verbose, event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'missed-shot':
                    toi, player_stats = process_missed_shot(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'blocked-shot':
                    toi, player_stats = process_blocked_shot(event, period_cd, compare_shift, away_players, home_players, away_players_sorted, home_players_sorted, last_event, game_time_event)
                elif event_details['event_name'] == 'penalty':
                    toi, player_stats = process_penalty(config.verbose, event, period_cd, compare_shift, away_players_sorted, home_players_sorted, last_event, game_time_event)
                elif event_details['event_name'] == 'stoppage':
                    toi, player_stats = process_stoppage(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'period-end':
                    toi, player_stats = process_period_end(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'delayed-penalty':
                    toi, player_stats = process_delayed_penalty(event, period_cd, compare_shift, away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'penalty-shot-missed':
                    toi, player_stats = process_penalty_shot(event, period_cd, compare_shift, away_players_sorted, home_players_sorted, last_event, game_time_event)
                else:
                    cwc = 0

                if sum(toi) >=0 or period_cd == 2:
                    game_id.append(game[i_event]['game_id'])
                    game_date.append(data_games[i_game]['game_date'])
                    away_teams.append(away_team)
                    home_teams.append(home_team)
                    period_id.append(event['period'])
                    period_code.append(period_cd)
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
            'period_code': period_code,
            'time_index': time_index,
            'time_on_ice': toi_list,
            'event_id': event_id,
            'shift_id': shift_id,
            'away_empty_net': away_empty_net,
            'home_empty_net': home_empty_net,
            'away_skaters': away_skaters,
            'home_skaters': home_skaters,
            'player_data': player_data,
        }

        save_game_data(data, config.file_paths["game_output_pkl"] + f'{str(game_id[0])}')

        del data['player_data']
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

        attributes_to_sum = ['goal', 'assist', 'shot_on_goal', 'goal_against', 'penalties_duration', 'hit_another_player', 'hit_by_player', 'giveaways', 'takeaways']  # Example attributes

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
            cols_of_interest = df[away_columns[attr]]
            # 2. For each column, gather the non-NaN lists, then sum them up.
            #    This will give you a Series of "vector sums" (one for each column).
            vector_sums_per_column = cols_of_interest.apply(
                lambda col: np.array(col.dropna().tolist()).sum(axis=0)  # sum across rows
            )
            combined_sum = vector_sums_per_column.to_numpy().sum(axis=1)
            team_sums['away'][f'away_{attr}_sum'] = combined_sum.tolist()

            # Sum for home team
            cols_of_interest = df[home_columns[attr]]
            # 2. For each column, gather the non-NaN lists, then sum them up.
            #    This will give you a Series of "vector sums" (one for each column).
            vector_sums_per_column = cols_of_interest.apply(
                lambda col: np.array(col.dropna().tolist()).sum(axis=0)  # sum across rows
            )
            combined_sum = vector_sums_per_column.to_numpy().sum(axis=1)
            team_sums['home'][f'home_{attr}_sum'] = combined_sum.tolist()
        #     sum_series = df[home_columns[attr]].sum().to_list()
        #     sum_list = [int(x) for x in sum_series]
        #     team_sums['home'][f'home_{attr}_sum'] = sum(sum_list)

        if (team_sums['home']['home_goal_sum'][2] > 0) or (team_sums['away']['away_goal_sum'][2] > 0):
            if team_sums['home']['home_goal_sum'][2] > team_sums['away']['away_goal_sum'][2]:
                team_sums['away']['away_goal_sum'][2] = 0
                team_sums['home']['home_goal_sum'][2] = 1
            elif team_sums['home']['home_goal_sum'][2] < team_sums['away']['away_goal_sum'][2]:
                team_sums['away']['away_goal_sum'][2] = 1
                team_sums['home']['home_goal_sum'][2] = 0

        # print(f'game_id: {game_id[0]}  toi: {sum(df["time_on_ice"].sum())} {data_games[i_game]["playbyplay"]}')
        all_good = True
        reasons = []
        reason = ''
        if data_games[i_game]['away_goals'] != sum(team_sums['away']['away_goal_sum']):
            all_good = False
            reason = 'away_goals'
            reasons.append(reason)
        if data_games[i_game]['home_goals'] != sum(team_sums['home']['home_goal_sum']):
            all_good = False
            reason = 'home_goals'
            reasons.append(reason)
        if data_games[i_game]['away_sog'] != sum(team_sums['away']['away_shot_on_goal_sum'][0:2]):
            all_good = False
            reason = 'away_sog'
            reasons.append(reason)
        if data_games[i_game]['home_sog'] != sum(team_sums['home']['home_shot_on_goal_sum'][0:2]):
            all_good = False
            reason = 'home_sog'
            reasons.append(reason)
        if data_games[i_game]['away_pim'] != sum(team_sums['away']['away_penalties_duration_sum']):
            all_good = False
            reason = 'away_pim'
            reasons.append(reason)
        if data_games[i_game]['home_pim'] != sum(team_sums['home']['home_penalties_duration_sum']):
            all_good = False
            reason = 'home_pim'
            reasons.append(reason)
        if data_games[i_game]['away_hits'] != sum(team_sums['away']['away_hit_another_player_sum']):
            all_good = False
            reason = 'away_hits'
            reasons.append(reason)
        if data_games[i_game]['home_hits'] != sum(team_sums['home']['home_hit_another_player_sum']):
            all_good = False
            reason = 'home_hits'
            reasons.append(reason)
        if data_games[i_game]['away_hits'] != sum(team_sums['home']['home_hit_by_player_sum']):
            all_good = False
            reason = 'away_hits_v2'
            reasons.append(reason)
        if data_games[i_game]['home_hits'] != sum(team_sums['away']['away_hit_by_player_sum']):
            all_good = False
            reason = 'home_hits_v2'
            reasons.append(reason)
        if data_games[i_game]['away_give'] != sum(team_sums['away']['away_giveaways_sum']):
            all_good = False
            reason = 'away_giveaways'
            reasons.append(reason)
        if data_games[i_game]['home_give'] != sum(team_sums['home']['home_giveaways_sum']):
            all_good = False
            reason = 'home_giveaways'
            reasons.append(reason)
        if data_games[i_game]['away_take'] != sum(team_sums['away']['away_takeaways_sum']):
            all_good = False
            reason = 'away_takeaways'
            reasons.append(reason)
        if data_games[i_game]['home_take'] != sum(team_sums['home']['home_takeaways_sum']):
            all_good = False
            reason = 'home_takeaways'
            reasons.append(reason)


        if not all_good:
            print(f'reasons: {reasons}')
            print(f'shift data: {team_sums["away"]}  {team_sums["home"]}')
            print(f'away_goals {data_games[i_game]["away_goals"]} away_sog {data_games[i_game]["away_sog"]} away_pim {data_games[i_game]["away_pim"]} away_takeaways {data_games[i_game]["away_take"]} away_giveaways {data_games[i_game]["away_give"]}')
            print(f'home_goals {data_games[i_game]["home_goals"]} home_sog {data_games[i_game]["home_sog"]} home_pim {data_games[i_game]["home_pim"]} home_takeaways {data_games[i_game]["home_take"]} home_giveaways {data_games[i_game]["home_give"]}')
        else:
            pass
            # print(f'game totals confirmed')

        print('\n')
        # Step 4: Export the DataFrame to CSV

        df.to_csv(config.file_paths['game_output_csv'] + f'{str(game_id[0])}.csv', na_rep='', index=False)

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


def process_faceoff(event, period_code, compare_shift, away_players, home_players):
    player_stats = []
    toi = [0, 0, 0]
    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        if (event['faceoff_winner'] == player_id['player_id']) or (event['faceoff_loser'] == player_id['player_id']):
            away_player_stats['faceoff_taken'][period_code] += 1
            if event['faceoff_winner'] == player_id['player_id']:
                away_player_stats['faceoff_won'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        if (event['faceoff_winner'] == player_id['player_id']) or (event['faceoff_loser'] == player_id['player_id']):
            home_player_stats['faceoff_taken'][period_code] += 1
            if event['faceoff_winner'] == player_id['player_id']:
                home_player_stats['faceoff_won'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_hit(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        if event['hitting_player'] == player_id['player_id']:
            away_player_stats['hit_another_player'][period_code] += 1
        if event['hittee_player'] == player_id['player_id']:
            away_player_stats['hit_by_player'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        if event['hitting_player'] == player_id['player_id']:
            home_player_stats['hit_another_player'][period_code] += 1
        if event['hittee_player'] == player_id['player_id']:
            home_player_stats['hit_by_player'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_takeaway(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        if event['takeaway'] == player_id['player_id']:
            away_player_stats['takeaways'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        if event['takeaway'] == player_id['player_id']:
            home_player_stats['takeaways'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_giveaway(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        if event['giveaway'] == player_id['player_id']:
            away_player_stats['giveaways'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        if event['giveaway'] == player_id['player_id']:
            home_player_stats['giveaways'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_goal(event, period_code, compare_shift, away_players, home_players, away_players_sorted, home_players_sorted, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    on_ice = [int(sweater[0]) for sweater in compare_shift['away_players']]
    for player_id in away_players_sorted.keys():
        player_data = away_players_sorted[player_id]
        away_player_stats = create_player_stats(player_data)
        if away_players_sorted[player_id]['player_sweater'] in on_ice:
            away_player_stats['toi'][period_code] += toi[period_code]
        if event['shot_attempt'] == player_id:
            away_player_stats['shot_attempt'][period_code] += 1
        if event['shot_on_goal'] == player_id:
            away_player_stats['shot_on_goal'][period_code] += 1
        if event['goal'] == player_id:
            away_player_stats['goal'][period_code] += 1
        if event['goal_assist1'] == player_id:
            away_player_stats['assist'][period_code] += 1
        if event['goal_assist2'] == player_id:
            away_player_stats['assist'][period_code] += 1
        if event['goal_against'] == player_id:
            away_player_stats['goal_against'][period_code] += 1

        player_stats.append(away_player_stats)

    on_ice = [int(sweater[0]) for sweater in compare_shift['home_players']]
    for player_id in home_players_sorted.keys():
        player_data = home_players_sorted[player_id]
        home_player_stats = create_player_stats(player_data)
        if home_players_sorted[player_id]['player_sweater'] in on_ice:
            home_player_stats['toi'][period_code] += toi[period_code]
        if event['shot_attempt'] == player_id:
            home_player_stats['shot_attempt'][period_code] += 1
        if event['shot_on_goal'] == player_id:
            home_player_stats['shot_on_goal'][period_code] += 1
        if event['goal'] == player_id:
            home_player_stats['goal'][period_code] += 1
        if event['goal_assist1'] == player_id:
            home_player_stats['assist'][period_code] += 1
        if event['goal_assist2'] == player_id:
            home_player_stats['assist'][period_code] += 1
        if event['goal_against'] == player_id:
            home_player_stats['goal_against'][period_code] += 1

        player_stats.append(home_player_stats)

    return toi, player_stats


def process_shot_on_goal(verbose, event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        if event['shot_attempt'] == player_id['player_id']:
            away_player_stats['shot_attempt'][period_code] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            if verbose:
                print(f'away: adding shot for {player_id["player_lname"]}')
            away_player_stats['shot_on_goal'][period_code] += 1
        if event['shot_saved'] == player_id['player_id']:
            away_player_stats['shot_saved'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        if event['shot_attempt'] == player_id['player_id']:
            home_player_stats['shot_attempt'][period_code] += 1
        if event['shot_on_goal'] == player_id['player_id']:
            if verbose:
                print(f'home: adding shot for {player_id["player_lname"]}')
            home_player_stats['shot_on_goal'][period_code] += 1
        if event['shot_saved'] == player_id['player_id']:
            home_player_stats['shot_saved'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_missed_shot(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        if event['missed_shot_attempt'] == player_id['player_id']:
            away_player_stats['shot_missed'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        if event['missed_shot_attempt'] == player_id['player_id']:
            home_player_stats['shot_missed'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_blocked_shot(event, period_code, compare_shift, away_players, home_players, away_players_sorted, home_players_sorted, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    on_ice = [int(sweater[0]) for sweater in compare_shift['away_players']]
    for player_id in away_players_sorted.keys():
        player_data = away_players_sorted[player_id]
        away_player_stats = create_player_stats(player_data)
        if away_players_sorted[player_id]['player_sweater'] in on_ice:
            away_player_stats['toi'][period_code] += toi[period_code]
        if event['blocked_shot_attempt'] == player_id:
            away_player_stats['shot_attempt'][period_code] += 1
        if event['blocked_shot_saved'] == player_id:
            away_player_stats['shot_blocked'][period_code] += 1
        player_stats.append(away_player_stats)


    on_ice = [int(sweater[0]) for sweater in compare_shift['home_players']]
    for player_id in home_players_sorted.keys():
        player_data = home_players_sorted[player_id]
        home_player_stats = create_player_stats(player_data)
        if home_players_sorted[player_id]['player_sweater'] in on_ice:
            home_player_stats['toi'][period_code] += toi[period_code]
        if event['blocked_shot_attempt'] == player_id:
            home_player_stats['shot_attempt'][period_code] += 1
        if event['blocked_shot_saved'] == player_id:
            home_player_stats['shot_blocked'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_penalty(verbose, event, period_code, compare_shift, away_players_sorted, home_players_sorted, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    on_ice = [int(sweater[0]) for sweater in compare_shift['away_players']]
    for player_id in away_players_sorted.keys():
        player_data = away_players_sorted[player_id]
        away_player_stats = create_player_stats(player_data)
        if away_players_sorted[player_id]['player_sweater'] in on_ice:
            away_player_stats['toi'][period_code] += toi[period_code]
        if event['penalty_served'] == player_id:
            away_player_stats['penalties_served'][period_code] += 1
        if event['penalty_committed'] == player_id:
            away_player_stats['penalties'][period_code] += 1
        if event['penalty_committed'] is None:
            if event['penalty_served'] == player_id:
                away_player_stats['penalties_duration'][period_code] += event['penalty_duration']
                if verbose:
                    print(f'away: adding {event["penalty_duration"]} minutes for {player_data["player_lname"]}')
        else:
            if event['penalty_committed'] == player_id:
                away_player_stats['penalties_duration'][period_code] += event['penalty_duration']
                if verbose:
                    print(f'away: adding {event["penalty_duration"]} minutes for {player_data["player_lname"]}')
        if event['penalty_drawn'] == player_id:
            away_player_stats['penalties_drawn'][period_code] += 1

        player_stats.append(away_player_stats)

    on_ice = [int(sweater[0]) for sweater in compare_shift['home_players']]
    for player_id in home_players_sorted.keys():
        player_data = home_players_sorted[player_id]
        home_player_stats = create_player_stats(player_data)
        if home_players_sorted[player_id]['player_sweater'] in on_ice:
            home_player_stats['toi'][period_code] += toi[period_code]
        if event['penalty_served'] == player_id:
            home_player_stats['penalties_served'][period_code] += 1
        if event['penalty_committed'] == player_id:
            home_player_stats['penalties'][period_code] += 1
        if event['penalty_committed'] is None:
            if event['penalty_served'] == player_id:
                home_player_stats['penalties_duration'][period_code] += event['penalty_duration']
                if verbose:
                    print(f'home: adding {event["penalty_duration"]} minutes for {player_data["player_lname"]}')
        else:
            if event['penalty_committed'] == player_id:
                home_player_stats['penalties_duration'][period_code] += event['penalty_duration']
                if verbose:
                    print(f'home: adding {event["penalty_duration"]} minutes for {player_data["player_lname"]}')
        if event['penalty_drawn'] == player_id:
            home_player_stats['penalties_drawn'][period_code] += 1

        player_stats.append(home_player_stats)

    return toi, player_stats


def process_penalty_shot(event, period_code, compare_shift, away_players_sorted, home_players_sorted, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]

    for player_id in away_players_sorted.keys():
        player_data = away_players_sorted[player_id]
        away_player_stats = create_player_stats(player_data)
        if event['penalty_shot'] == player_id:
            away_player_stats['penalty_shot'][period_code] += 1
        if event['penalty_shot_goal'] == player_id:
            away_player_stats['penalty_shot_goal'][period_code] += 1
        if event['penalty_shot_saved'] == player_id:
            away_player_stats['penalty_shot_saved'][period_code] += 1

        player_stats.append(away_player_stats)

    for player_id in home_players_sorted.keys():
        player_data = home_players_sorted[player_id]
        home_player_stats = create_player_stats(player_data)
        if event['penalty_shot'] == player_id:
            home_player_stats['penalty_shot'][period_code] += 1
        if event['penalty_shot_goal'] == player_id:
            home_player_stats['penalty_shot_goal'][period_code] += 1
        if event['penalty_shot_saved'] == player_id:
            home_player_stats['penalty_shot_saved'][period_code] += 1

        player_stats.append(home_player_stats)

    return toi, player_stats


def process_miss(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        if event['miss'] == player_id['player_id']:
            away_player_stats['shot_attempt'][period_code] += 1
        if event['miss'] == player_id['player_id']:
            away_player_stats['shot_missed'][period_code] += 1
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        if event['miss'] == player_id['player_id']:
            home_player_stats['shot_attempt'][period_code] += 1
        if event['miss'] == player_id['player_id']:
            home_player_stats['shot_missed'][period_code] += 1
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_stoppage(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
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
                                 'rink-repair', 'official-injury', 'premature-substitution', 'chlg-league-off-side',
                                 'switch-sides']:
            pass  # data lacks detail to specify which goalie / team
        else:
            print(f'away stoppage reason: {event["stoppage"]}')
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
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
                                 'rink-repair', 'official-injury', 'premature-substitution', 'chlg-league-off-side',
                                 'switch-sides']:
            pass  # data lacks detail to specify which goalie / team
        else:
             print(f'home stoppage reason: {event["stoppage"]}')
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_period_end(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_game_end(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        player_stats.append(home_player_stats)

    return toi, player_stats


def process_delayed_penalty(event, period_code, compare_shift, away_players, home_players, last_event, game_time_event):
    player_stats = []
    toi = [0, 0, 0]
    toi[period_code] = calc_toi(game_time_event, last_event)

    for away_player in compare_shift['away_players']:
        player_id = away_players[int(away_player[0])]
        away_player_stats = create_player_stats(player_id)
        away_player_stats['toi'][period_code] += toi[period_code]
        player_stats.append(away_player_stats)

    for home_player in compare_shift['home_players']:
        player_id = home_players[int(home_player[0])]
        home_player_stats = create_player_stats(player_id)
        home_player_stats['toi'][period_code] += toi[period_code]
        player_stats.append(home_player_stats)

    return toi, player_stats


def calc_toi(game_time_event, last_event):
    if last_event is None:
        return -1
    last_game_time_event = period_time_to_game_time(last_event['period'], last_event['game_time'])
    toi = game_time_event - last_game_time_event
    return toi
