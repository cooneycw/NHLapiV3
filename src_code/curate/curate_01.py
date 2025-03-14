import multiprocessing as mp
from multiprocessing import Queue
from src_code.utils.utils import (
    period_time_to_game_time,
    create_roster_dicts,
    create_ordered_roster,
    create_player_stats,
    save_game_data
)
from typing import Dict, List, Any
import copy
import heapq
import numpy as np
import pandas as pd


def curate_data(config):
    """Main function for parallel game data curation with incremental updates."""
    import os
    import pickle

    print("Gathering game data for curation...")
    # Load data
    dimension_names = "all_names"
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"
    dimension_games = "all_boxscores"
    dimension_players = "all_players"
    dimension_curated = "all_curated"  # New dimension for curated games

    # Check if we need to do a full reload or incremental update
    do_full_reload = getattr(config, "reload_curate", False)

    if not do_full_reload:
        processed_games_list, _ = config.load_data(dimension_curated)
        if processed_games_list is None:
            processed_games = set()
            print("No valid cached curation data. Processing all games.")
        else:
            # Convert the sorted list back to a set for efficient lookups
            processed_games = set(processed_games_list)
            print(f"Using cached curation data ({len(processed_games)} games)...")
    else:
        print("Reload requested. Reprocessing all games.")
        processed_games = set()

    # Load data
    data_names = config.load_data(dimension_names)
    data_games = config.load_data(dimension_games)
    data_players = config.load_data(dimension_players)
    data_shifts = config.load_data(dimension_shifts)
    data_plays = config.load_data(dimension_plays)
    data_game_roster = config.load_data(dimension_game_rosters)

    # Identify games in each dataset
    play_game_ids = set()
    for i, plays in enumerate(data_plays):
        if plays and len(plays) > 0 and 'game_id' in plays[0]:
            play_game_ids.add(plays[0]['game_id'])

    boxscore_game_ids = set()
    for boxscore in data_games:
        if 'id' in boxscore:
            boxscore_game_ids.add(boxscore['id'])

    shift_game_ids = set()
    for shifts in data_shifts:
        if shifts and len(shifts) > 0 and 'game_id' in shifts[0]:
            shift_game_ids.add(shifts[0]['game_id'])

    roster_game_ids = set()
    for roster in data_game_roster:
        if roster and len(roster) > 0 and 'game_id' in roster[0]:
            roster_game_ids.add(roster[0]['game_id'])

    # Find games that have data in all datasets
    available_games = play_game_ids.intersection(boxscore_game_ids, shift_game_ids, roster_game_ids)

    # Filter out games already processed unless doing a full reload
    if do_full_reload:
        games_to_process = available_games
    else:
        games_to_process = available_games - processed_games

    if not games_to_process:
        print("No new games to process for curation.")
        return

    print(f"Found {len(games_to_process)} games that need curation...")

    # Create mappings from game_id to data index
    play_map = {}
    for i, plays in enumerate(data_plays):
        if plays and len(plays) > 0 and 'game_id' in plays[0]:
            play_map[plays[0]['game_id']] = i

    game_map = {}
    for i, game in enumerate(data_games):
        if 'id' in game:
            game_map[game['id']] = i

    shift_map = {}
    for i, shifts in enumerate(data_shifts):
        if shifts and len(shifts) > 0 and 'game_id' in shifts[0]:
            shift_map[shifts[0]['game_id']] = i

    roster_map = {}
    for i, roster in enumerate(data_game_roster):
        if roster and len(roster) > 0 and 'game_id' in roster[0]:
            roster_map[roster[0]['game_id']] = i

    # Create process arguments for each game to process
    process_args = []
    for game_id in sorted(games_to_process):
        if (game_id in play_map and game_id in game_map and
                game_id in shift_map and game_id in roster_map):
            process_args.append((
                play_map[game_id],
                data_plays[play_map[game_id]],
                config,
                data_games[game_map[game_id]],
                data_shifts[shift_map[game_id]],
                data_game_roster[roster_map[game_id]],
                None  # Will be replaced with the queue in the multiprocessing section
            ))
        else:
            print(f"Warning: Missing data for game {game_id}, skipping.")

    # Verify we have games to process
    if not process_args:
        print("No games with complete data to process.")
        return

    print(f"Processing {len(process_args)} games...")

    # Create manager for shared resources
    with mp.Manager() as manager:
        # Create managed queue
        result_queue = manager.Queue()

        # Update process args with the queue
        process_args = [(args[0], args[1], args[2], args[3], args[4], args[5], result_queue)
                        for args in process_args]

        # Start reporter process
        reporter_process = mp.Process(
            target=validation_reporter,
            args=(result_queue, len(process_args))
        )
        reporter_process.start()

        # Create pool of workers
        with mp.Pool(processes=max(1, int(0.35 * config.max_workers))) as pool:
            # Process games in parallel
            pool.starmap(process_single_game, process_args)

            # Signal reporter that all games are processed
            result_queue.put(None)  # Sentinel value

            # Wait for reporter to finish
            reporter_process.join()

    # Update processed games list with newly processed games
    newly_processed = set()
    for args in process_args:
        game = args[1]  # data_plays for this game
        if game and len(game) > 0 and 'game_id' in game[0]:
            newly_processed.add(game[0]['game_id'])

    processed_games.update(newly_processed)
    # After processing games, save the updated processed_games
    config.save_data(dimension_curated, processed_games)
    print(f"Completed processing {len(newly_processed)} new games for curation.")


def process_single_game(game_index: int,
                        game: Dict,
                        config: Any,
                        data_games: List[Dict],
                        data_shifts: List[Dict],
                        data_game_roster: List[Dict],
                        queue: Queue) -> None:
    """Process a single game and put validation results in the queue."""
    try:
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

        away_team = data_games[game_index]['awayTeam']
        home_team = data_games[game_index]['homeTeam']
        away_players, home_players = create_roster_dicts(data_game_roster[game_index], away_team, home_team)
        away_players_sorted, home_players_sorted = create_ordered_roster(data_game_roster[game_index], away_team,
                                                                         home_team)
        last_event = None

        for i_event, event in enumerate(game):
            event_details = config.event_categ.get(event['event_code'])
            if event_details is None:
                continue
            if not event_details['sport_stat']:
                continue

            game_time_event = period_time_to_game_time(event['period'], event['game_time'])

            while True:
                compare_shift = data_shifts[game_index][i_shift]
                shift_details = config.shift_categ.get(compare_shift['event_type'])

                if shift_details is None:
                    i_shift += 1
                    continue
                if not shift_details['sport_stat']:
                    i_shift += 1
                    continue
                break

            game_time_shift = period_time_to_game_time(int(compare_shift['period']), compare_shift['game_time'])
            period_cd = 0
            if event['overtime']:
                period_cd = 1
            elif event['shootout']:
                period_cd = 2

            if (((event_details['event_name'] == shift_details['shift_name']) and (
                    game_time_event == game_time_shift)) or
                    ((event_details['event_name'] == 'penalty-shot') and (game_time_event == game_time_shift)) or
                    ((event_details['event_name'] == 'penalty-shot-missed') and (game_time_event == game_time_shift))):

                empty_net_data = process_empty_net(compare_shift)

                # Process different event types using existing processing functions
                if event_details['event_name'] == 'faceoff':
                    toi, player_stats = process_faceoff(event, period_cd, compare_shift, away_players, home_players)
                elif event_details['event_name'] == 'hit':
                    toi, player_stats = process_hit(event, period_cd, compare_shift, away_players, home_players,
                                                    last_event, game_time_event)
                elif event_details['event_name'] == 'giveaway':
                    toi, player_stats = process_giveaway(event, period_cd, compare_shift, away_players, home_players,
                                                         last_event, game_time_event)
                elif event_details['event_name'] == 'takeaway':
                    toi, player_stats = process_takeaway(event, period_cd, compare_shift, away_players, home_players,
                                                         last_event, game_time_event)
                elif event_details['event_name'] == 'goal':
                    toi, player_stats = process_goal(event, period_cd, compare_shift, away_players, home_players,
                                                     away_players_sorted, home_players_sorted, last_event,
                                                     game_time_event)
                elif event_details['event_name'] == 'shot-on-goal':
                    toi, player_stats = process_shot_on_goal(config.verbose, event, period_cd, compare_shift,
                                                             away_players, home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'missed-shot':
                    toi, player_stats = process_missed_shot(event, period_cd, compare_shift, away_players, home_players,
                                                            last_event, game_time_event)
                elif event_details['event_name'] == 'blocked-shot':
                    toi, player_stats = process_blocked_shot(event, period_cd, compare_shift, away_players,
                                                             home_players, away_players_sorted, home_players_sorted,
                                                             last_event, game_time_event)
                elif event_details['event_name'] == 'penalty':
                    toi, player_stats = process_penalty(config.verbose, event, period_cd, compare_shift,
                                                        away_players_sorted, home_players_sorted, last_event,
                                                        game_time_event)
                elif event_details['event_name'] == 'stoppage':
                    toi, player_stats = process_stoppage(event, period_cd, compare_shift, away_players, home_players,
                                                         last_event, game_time_event)
                elif event_details['event_name'] == 'period-end':
                    toi, player_stats = process_period_end(event, period_cd, compare_shift, away_players, home_players,
                                                           last_event, game_time_event)
                elif event_details['event_name'] == 'delayed-penalty':
                    toi, player_stats = process_delayed_penalty(event, period_cd, compare_shift, away_players,
                                                                home_players, last_event, game_time_event)
                elif event_details['event_name'] == 'penalty-shot-missed':
                    toi, player_stats = process_penalty_shot(event, period_cd, compare_shift, away_players_sorted,
                                                             home_players_sorted, last_event, game_time_event)

                if sum(toi) >= 0 or period_cd == 2:
                    game_id.append(game[i_event]['game_id'])
                    game_date.append(data_games[game_index]['game_date'])
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

        # Create game data dictionary
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

        # Save game data
        save_game_data(data, config.file_paths["game_output_pkl"] + f'{str(game_id[0])}')

        # Create DataFrame
        df_data = data.copy()
        del df_data['player_data']
        df = pd.DataFrame(df_data)

        player_attributes = list(player_data[0][0].keys())
        standardized_players = []

        # Sort players by sweater number
        for player_id in sorted(away_players.keys()):
            standardized_players.append(away_players[player_id])
        for player_id in sorted(home_players.keys()):
            standardized_players.append(home_players[player_id])

        num_players = len(standardized_players)

        # Initialize new columns
        new_columns = {'player_data': player_data}
        new_columns.update({
            f'player_{i}_{attr}': pd.NA
            for i in range(1, num_players + 1)
            for attr in player_attributes
        })

        # Create DataFrame with new columns
        new_columns_df = pd.DataFrame(new_columns, index=df.index)

        # Create player ID to position mapping
        player_id_to_position = {
            player['player_id']: idx
            for idx, player in enumerate(standardized_players, start=1)
        }

        # Populate player columns
        def populate_player_columns(row):
            player_data_row = row['player_data']
            for player_in_row in player_data_row:
                player_id = player_in_row['player_id']
                if player_id in player_id_to_position:
                    pos = player_id_to_position[player_id]
                    for attr in player_attributes:
                        col_name = f'player_{pos}_{attr}'
                        row[col_name] = player_in_row.get(attr, pd.NA)
            return row

        new_columns_df = new_columns_df.apply(populate_player_columns, axis=1)
        new_columns_df = new_columns_df.drop(columns=['player_data'])
        df = pd.concat([df, new_columns_df], axis=1)

        # Calculate team sums
        attributes_to_sum = [
            'goal', 'assist', 'shot_on_goal', 'goal_against',
            'penalties_duration', 'hit_another_player', 'hit_by_player',
            'giveaways', 'takeaways'
        ]

        away_columns = {attr: [] for attr in attributes_to_sum}
        home_columns = {attr: [] for attr in attributes_to_sum}

        for attr in attributes_to_sum:
            away_columns[attr] = [f'player_{i}_{attr}' for i in range(1, len(away_players) + 1)]
            home_columns[attr] = [f'player_{i}_{attr}' for i in range(len(away_players) + 1, 2 * len(away_players) + 1)]

        team_sums = {'away': {}, 'home': {}}

        for attr in attributes_to_sum:
            # Calculate away team sums
            cols_of_interest = df[away_columns[attr]]
            vector_sums = cols_of_interest.apply(
                lambda col: np.array(col.dropna().tolist()).sum(axis=0)
            )
            team_sums['away'][f'away_{attr}_sum'] = vector_sums.to_numpy().sum(axis=1).tolist()

            # Calculate home team sums
            cols_of_interest = df[home_columns[attr]]
            vector_sums = cols_of_interest.apply(
                lambda col: np.array(col.dropna().tolist()).sum(axis=0)
            )
            team_sums['home'][f'home_{attr}_sum'] = vector_sums.to_numpy().sum(axis=1).tolist()

        # Handle shootout goals
        if (team_sums['home']['home_goal_sum'][2] > 0) or (team_sums['away']['away_goal_sum'][2] > 0):
            if team_sums['home']['home_goal_sum'][2] > team_sums['away']['away_goal_sum'][2]:
                team_sums['away']['away_goal_sum'][2] = 0
                team_sums['home']['home_goal_sum'][2] = 1
            elif team_sums['home']['home_goal_sum'][2] < team_sums['away']['away_goal_sum'][2]:
                team_sums['away']['away_goal_sum'][2] = 1
                team_sums['home']['home_goal_sum'][2] = 0

        # Create validation result
        validation_result = create_validation_result(
            game_id=game_id[0],
            team_sums=team_sums,
            game_data=data_games[game_index],
            toi_sum=sum(df["time_on_ice"].sum())
        )

        # Save to CSV
        if config.produce_csv:
            df.to_csv(config.file_paths['game_output_csv'] + f'{str(game_id[0])}.csv', na_rep='', index=False)

        # Put validation result in queue
        queue.put(validation_result)

    except Exception as e:
        # Handle any errors during processing
        error_result = {
            'game_id': game[0]['game_id'] if game else None,
            'all_good': False,
            'reasons': [{
                'type': 'processing_error',
                'expected': 'successful processing',
                'actual': str(e),
                'difference': 'error occurred'
            }],
            'team_sums': None,
            'game_data': data_games[game_index] if game_index < len(data_games) else None,
            'toi_sum': None
        }
        queue.put(error_result)


def create_validation_result(game_id: int,
                             team_sums: Dict,
                             game_data: Dict,
                             toi_sum: float) -> Dict[str, Any]:
    """Create a validation result dictionary with difference magnitudes."""
    all_good = True
    reasons = []

    # Validation checks with difference tracking
    checks = [
        ('away_goals', sum(team_sums['away']['away_goal_sum']), game_data['away_goals']),
        ('home_goals', sum(team_sums['home']['home_goal_sum']), game_data['home_goals']),
        ('away_sog', sum(team_sums['away']['away_shot_on_goal_sum'][0:2]), game_data['away_sog']),
        ('home_sog', sum(team_sums['home']['home_shot_on_goal_sum'][0:2]), game_data['home_sog']),
        ('away_pim', sum(team_sums['away']['away_penalties_duration_sum']), game_data['away_pim']),
        ('home_pim', sum(team_sums['home']['home_penalties_duration_sum']), game_data['home_pim']),
        ('away_hits', sum(team_sums['away']['away_hit_another_player_sum']), game_data['away_hits']),
        ('home_hits', sum(team_sums['home']['home_hit_another_player_sum']), game_data['home_hits']),
        ('away_hits_v2', sum(team_sums['home']['home_hit_by_player_sum']), game_data['away_hits']),
        ('home_hits_v2', sum(team_sums['away']['away_hit_by_player_sum']), game_data['home_hits']),
        ('away_giveaways', sum(team_sums['away']['away_giveaways_sum']), game_data['away_give']),
        ('home_giveaways', sum(team_sums['home']['home_giveaways_sum']), game_data['home_give']),
        ('away_takeaways', sum(team_sums['away']['away_takeaways_sum']), game_data['away_take']),
        ('home_takeaways', sum(team_sums['home']['home_takeaways_sum']), game_data['home_take'])
    ]

    for stat_name, actual, expected in checks:
        if actual != expected:
            all_good = False
            reasons.append({
                'type': stat_name,
                'expected': expected,
                'actual': actual,
                'difference': actual - expected
            })

    return {
        'game_id': game_id,
        'all_good': all_good,
        'reasons': reasons,
        'team_sums': team_sums,
        'game_data': game_data,
        'toi_sum': toi_sum
    }


def validation_reporter(result_queue: Queue, total_games: int) -> None:
    """
    Collect and report validation results in game_id order.
    Uses a min heap to maintain ordering.
    """
    results_heap = []  # Min heap for ordering results
    games_processed = 0
    next_game_to_report = None

    while True:
        # Get result from queue
        result = result_queue.get()

        # Check for sentinel value indicating completion
        if result is None:
            # Process any remaining results in the heap
            while results_heap:
                _, result = heapq.heappop(results_heap)
                report_result(result)
            break

        # Add to heap
        heapq.heappush(results_heap, (result['game_id'], result))
        games_processed += 1

        # Process results in order
        while results_heap and (next_game_to_report is None or
                                results_heap[0][0] == next_game_to_report):
            _, result = heapq.heappop(results_heap)
            report_result(result)

            # Update next expected game
            next_game_to_report = result['game_id'] + 1


def report_result(result):
    """Helper function to report a single game result."""
    if not result:
        print("Warning: Null result received in report_result")
        return

    if 'team_sums' not in result or result['team_sums'] is None:
        game_id = result.get('game_id', 'unknown')
        print(f"\ngame_id: {game_id} - Missing team_sums data")
        return

    if not result['all_good']:
        # Print validation failure details
        print(
            f"\ngame_id: {result['game_id']}  toi: {result['toi_sum']} {result['game_data'].get('playbyplay', 'N/A')}")
        print("reasons:")
        for reason in result['reasons']:
            print(f"  {reason['type']}: expected {reason['expected']}, "
                  f"got {reason['actual']}, diff {reason['difference']}")

        try:
            print(f"shift data: {result['team_sums']['away']}  {result['team_sums']['home']}")
            print(f"away_goals {result['game_data']['away_goals']} "
                  f"away_sog {result['game_data']['away_sog']} "
                  f"away_pim {result['game_data']['away_pim']} "
                  f"away_takeaways {result['game_data']['away_take']} "
                  f"away_giveaways {result['game_data']['away_give']}")
            print(f"home_goals {result['game_data']['home_goals']} "
                  f"home_sog {result['game_data']['home_sog']} "
                  f"home_pim {result['game_data']['home_pim']} "
                  f"home_takeaways {result['game_data']['home_take']} "
                  f"home_giveaways {result['game_data']['home_give']}")
        except (KeyError, TypeError) as e:
            print(f"Error printing detailed results: {e}")

    elif result['game_id'] % 20 == 0:
        # Print success message for every 20th game
        print(
            f"\ngame_id: {result['game_id']}  toi: {result['toi_sum']} {result['game_data'].get('playbyplay', 'N/A')}")
        print("game totals confirmed")
        print("\n")


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
                                 'switch-sides', 'chlg-hm-puck-over-glass']:
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
                                 'switch-sides', 'chlg-hm-puck-over-glass']:
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