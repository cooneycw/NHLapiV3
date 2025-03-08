from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from src_code.utils.utils import create_dummy_player
import os
import re
import requests
import time


def get_season_data(config):
    print(f'Gathering season data...')
    dimension = "all_seasons"

    # Only delete files if explicitly requested
    # This selective approach avoids deleting everything unnecessarily
    if config.delete_files:
        file_path = config.file_paths[dimension]
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

    # Try to load existing data
    prior_data = config.load_data(dimension)

    # Check if we have valid data and don't need to reload
    if prior_data and not config.reload_seasons:
        print(f'Using cached season data ({len(prior_data)} seasons)...')
        # Use the cached data
        for season in prior_data:
            _ = config.Season(season)
    else:
        # We need to fetch new data
        reason = "Reload requested." if config.reload_seasons else "No valid cached data."
        print(f'{reason} Fetching season data from API...')

        final_url = config.get_endpoint("seasons")
        response = requests.get(final_url)
        season_data = response.json()

        # Register the seasons with the config object
        for season in season_data:
            _ = config.Season(season)

        # Save the data for future use
        config.save_data(dimension, season_data)

        print(f'Fetched and saved {len(season_data)} seasons.')


def get_team_list(config):
    print(f'Gathering team data...')
    dimension = "all_teams"

    # Handle selective file deletion if requested
    if config.delete_files:
        file_path = config.file_paths[dimension]
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

    # Try to load existing data
    prior_data = config.load_data(dimension)

    # Check if we have valid data and don't need to reload
    if prior_data and not config.reload_teams:
        print(f'Using cached team data ({len(prior_data)} teams)...')
        # Use the cached data
        for team in prior_data:
            _ = config.Team(team)
    else:
        # We need to fetch new data
        reason = "Reload requested." if config.reload_teams else "No valid cached data."
        print(f'{reason} Fetching team data from API...')

        final_url = config.get_endpoint("standings")
        response = requests.get(final_url)
        standings_data = response.json()

        # Extract and sort team abbreviations
        team_list = []
        for item in standings_data['standings']:
            team_abbrev = item['teamAbbrev']['default']
            team_list.append(team_abbrev)
        team_list = sorted(team_list)

        # Register the teams with the config object
        for team in team_list:
            config.Team(team)

        # Save the data for future use
        config.save_data(dimension, team_list)

        print(f'Fetched and saved {len(team_list)} teams.')


def fetch_games_for_team_and_season(config, season, team):
    print(f'Fetching games for {team[0]} in season {season[0]}...')

    final_url = config.get_endpoint("schedule", team=team[0], season=season[0])
    response = requests.get(final_url)
    response.raise_for_status()

    data = response.json()
    partial_data = []

    for game in data["games"]:
        # Skip non-regular-season games
        if game["gameType"] != 2:
            continue

        game_date = datetime.strptime(game["gameDate"], "%Y-%m-%d")
        home_team = game["homeTeam"]['abbrev']
        away_team = game["awayTeam"]['abbrev']

        # Skip future games for game object creation
        if game_date.date() >= config.curr_date:
            # Track future games with a status field
            future_game = (
                game['id'],
                game_date,
                away_team,
                home_team,
                season[0],
                'scheduled'
            )
            partial_data.append(future_game)
            continue

        # For past games, create the game object and add its tuple
        game_obj = config.Game.create_game(
            game['id'],
            game_date,
            home_team,
            away_team,
            season[0]
        )

        # If the game object is None, it means this game was already created
        # This is NORMAL BEHAVIOR, not an error - don't report as a warning
        if game_obj is None:
            # Game already exists in dictionary, just add the tuple to our list
            game_tuple = (game['id'], game_date, away_team, home_team, season[0])
            partial_data.append(game_tuple)
        else:
            # New game was created, add its tuple to our list
            partial_data.append(game_obj.get_game_tuple())

    return partial_data


def get_game_list(config):
    print('Gathering game data by team...')
    dimension = "all_games"

    # Handle selective file deletion if requested
    if config.delete_files:
        file_path = config.file_paths[dimension]
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")

    # Try to load existing data with detailed logging
    print(f"Attempting to load game data from {config.file_paths[dimension]}")
    prior_data = config.load_data(dimension)
    print(f"Load result: {'Data found with ' + str(len(prior_data)) + ' entries' if prior_data else 'No data found'}")

    # Check for the status update flag, defaulting to True if not set
    update_game_statuses = getattr(config, 'update_game_statuses', True)
    print(f"Config settings: reload_games={config.reload_games}, update_game_statuses={update_game_statuses}")

    # Check if we have valid data and don't need to reload
    if prior_data and not config.reload_games:
        print(f'Using cached game data ({len(prior_data)} games)...')

        # Split games into completed, scheduled and postponed
        completed_games = []
        future_games = []
        postponed_games = []
        failed_games = []
        games_needing_update = []

        # Process each game based on its status and date
        for game in prior_data:
            # Check if this is a game with status fields (new format)
            if len(game) > 5:
                status = game[5]
                if status == 'postponed':
                    # If date has passed and updates enabled, this game needs checking
                    if game[1].date() < config.curr_date and update_game_statuses:
                        games_needing_update.append(game)
                    else:
                        postponed_games.append(game)
                elif status == 'scheduled':
                    # If scheduled date has passed and updates enabled, check if it happened
                    if game[1].date() < config.curr_date and update_game_statuses:
                        games_needing_update.append(game)
                    else:
                        future_games.append(game)
                elif status in ['error', 'creation_failed']:
                    failed_games.append(game)
            # Standard format game tuple - assume it's a completed game
            elif game[1].date() < config.curr_date:
                completed_games.append(game)

        # Create game objects only for completed games
        print(f'Loading {len(completed_games)} completed games...')
        success_count = 0
        already_exists_count = 0
        error_count = 0

        for game in completed_games:
            try:
                game_obj = config.Game.create_game(game[0], game[1], game[2], game[3], game[4])
                if game_obj:
                    success_count += 1
                else:
                    # This is not a failure - the game already exists in the dictionary
                    already_exists_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error creating game object for {game[0]} - {game[2]} @ {game[3]}: {str(e)}")

        print(
            f"Successfully loaded {success_count} new games, {already_exists_count} were already in the dictionary, {error_count} errors")

        # Log about other game types
        if postponed_games:
            print(f'Found {len(postponed_games)} postponed games')
        if future_games:
            print(f'Found {len(future_games)} future scheduled games')
        if failed_games:
            print(f'Found {len(failed_games)} games that previously failed to create')

        # If we have games that need updates and updates are enabled
        if games_needing_update and update_game_statuses:
            print(
                f'Found {len(games_needing_update)} games that need updating (previously future/postponed, now past date)')

            # Identify which teams and seasons need updates
            teams_needing_update = set()
            seasons_needing_update = set()
            for game in games_needing_update:
                teams_needing_update.add(game[2])  # Away team
                teams_needing_update.add(game[3])  # Home team
                seasons_needing_update.add(game[4])  # Season

            print(
                f'Will update data for {len(teams_needing_update)} teams across {len(seasons_needing_update)} seasons')

            # Create a list of season-team pairs to update
            season_team_pairs = []
            for season in seasons_needing_update:
                for team in teams_needing_update:
                    season_obj = next(
                        (s for s in config.Season.get_selected_seasons(config.season_count) if s[0] == season), None)
                    team_obj = next((t for t in config.Team.get_teams() if t[0] == team), None)
                    if season_obj and team_obj:
                        season_team_pairs.append((season_obj, team_obj))

            print(f'Will fetch updates for {len(season_team_pairs)} season-team combinations')
        else:
            # No updates needed or updates disabled
            if games_needing_update and not update_game_statuses:
                print(f'Found {len(games_needing_update)} games that would need updating, but updates are disabled')

            # Nothing to update, return early
            print('No updates needed for game data')
            return
    else:
        # We need to fetch all game data
        reason = "Reload requested." if config.reload_games else "No valid cached data."
        print(f'{reason} Will fetch all game data from API...')

        # Get all season-team pairs for fetching
        season_team_pairs = []
        for season in config.Season.get_selected_seasons(config.season_count):
            for team in config.Team.get_teams():
                season_team_pairs.append((season, team))

        print(f'Will fetch data for {len(season_team_pairs)} season-team combinations')

    # At this point, season_team_pairs contains what we need to fetch
    print(f'Starting parallel fetch for {len(season_team_pairs)} season-team pairs...')

    # Set up our results array
    results_in_order = [None] * len(season_team_pairs)

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all jobs and track their start times
        future_to_index = {
            executor.submit(fetch_games_for_team_and_season, config, season, team): i
            for i, (season, team) in enumerate(season_team_pairs)
        }

        # Collect results as they complete
        complete_count = 0
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            try:
                data = future.result()
                results_in_order[i] = data

                complete_count += 1
                if complete_count % 10 == 0 or complete_count == len(season_team_pairs):
                    print(f'Processed {complete_count}/{len(season_team_pairs)} team-seasons...')
            except Exception as exc:
                season, team = season_team_pairs[i]
                print(f"Error fetching games for {team[0]} in season {season[0]}: {exc}")
                import traceback
                traceback.print_exc()

    # Collect all results
    new_games = []
    unique_game_ids = set()  # Track unique game IDs to handle duplicates
    duplicates = 0

    for games_list in results_in_order:
        if games_list:
            for game in games_list:
                game_id = game[0]
                if game_id not in unique_game_ids:
                    unique_game_ids.add(game_id)
                    new_games.append(game)
                else:
                    duplicates += 1

    print(f'Fetched {len(new_games)} unique games from API')
    if duplicates > 0:
        print(f'Filtered out {duplicates} duplicate games (expected due to team schedules overlap)')

    if prior_data and not config.reload_games and update_game_statuses:
        # This was a partial update - merge with existing data
        # Create a dictionary of game IDs to new game data
        new_games_dict = {game[0]: game for game in new_games}

        # Start with all existing games
        updated_games = []
        updated_count = 0

        for game in prior_data:
            game_id = game[0]
            # If we have updated data for this game, use it
            if game_id in new_games_dict:
                updated_games.append(new_games_dict[game_id])
                updated_count += 1
                # Remove from dict to track what we've processed
                del new_games_dict[game_id]
            else:
                # Keep existing data for games we didn't update
                updated_games.append(game)

        # Add any completely new games we found
        new_count = len(new_games_dict)
        for game in new_games_dict.values():
            updated_games.append(game)

        print(
            f'Updated game list: {updated_count} games updated, {new_count} new games added, {len(updated_games)} total games')
        all_games = updated_games
    else:
        # This was a full reload - just use the new data
        all_games = new_games
        print(f'New game data has {len(all_games)} total games')

    # Create game objects for completed games (exclude scheduled future games)
    completed_games = []
    for game in all_games:
        if len(game) <= 5:  # Standard format - assumed completed
            completed_games.append(game)
        elif game[5] not in ['scheduled', 'error', 'creation_failed']:  # Include completed games with status
            completed_games.append(game)

    print(f'Creating game objects for {len(completed_games)} completed games...')
    success_count = 0
    already_exists_count = 0
    error_count = 0

    for game in completed_games:
        try:
            if len(game) <= 5:  # Standard format
                game_obj = config.Game.create_game(game[0], game[1], game[2], game[3], game[4])
                if game_obj:
                    success_count += 1
                else:
                    # This is not a failure - it just means the game was already created
                    already_exists_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error creating game object for {game[0]} - {game[2]} @ {game[3]}: {str(e)}")

    print(
        f"Successfully created {success_count} new game objects, {already_exists_count} were already in the dictionary, {error_count} errors")

    # Save all game data for future use
    print(f'Saving {len(all_games)} total games to cache...')
    config.save_data(dimension, all_games)

    print('Game data gathering complete')


def process_game(game, config):
    """
    Process a single game: make the HTTP requests,
    process the boxscore, update game and player objects,
    and return lists of results.
    """
    # Skip processing for games beyond the current date.
    if game[1].date() >= config.curr_date:
        return None, None

    print(f'Getting boxscore for {game[0]}:{game[1]}:{game[2]}:{game[3]}')

    # Request boxscore data for version 1
    final_url_v1 = config.get_endpoint("boxscore_v1", game_id=game[0])
    response_v1 = requests.get(final_url_v1)
    response_v1.raise_for_status()  # Ensure the response is OK
    data_v1 = response_v1.json()

    # Request boxscore data for version 2
    final_url_v2 = config.get_endpoint("boxscore_v2", game_id=game[0])
    response_v2 = requests.get(final_url_v2)
    response_v2.raise_for_status()  # Ensure the response is OK
    data_v2 = response_v2.json()

    # Process the boxscore and update the game object.
    game_obj = config.Game.get_game(game[0])
    success, game_results = process_boxscore(game_obj.game_id, data_v1, data_v2)
    game_results_list = []
    player_results_list = []

    if success:
        game_results_list.append(game_results)
        game_obj.update_game(game_results)

        # Process player data for both teams
        for i, team in enumerate(['awayTeam', 'homeTeam']):
            position_data = data_v1['playerByGameStats'][team]
            for position in ['forwards', 'defense', 'goalies']:
                for player in position_data[position]:
                    # Process data based on the player position
                    if position != 'goalies':
                        player_results = process_forward_defense_data(player)
                    else:
                        player_results = process_goalie_data(player)
                    # Annotate the player results with additional info.
                    player_results['i'] = i
                    player_results['game'] = game

                    # Update the player object
                    player_obj = config.Player.create_player(player_results['player_id'])
                    player_obj.update_player(game[0], game[1], game[3 - i], game[4], player_results)
                    player_results_list.append(player_results)
    else:
        # Optionally, handle errors here (e.g. logging the failure)
        pass

    return game_results_list, player_results_list


def get_boxscore_list(config):
    print('Gathering boxscore data...')
    dimension_games = "all_boxscores"
    dimension_players = "all_players"
    prior_data_games = config.load_data(dimension_games)
    prior_data_players = config.load_data(dimension_players)

    if prior_data_games and prior_data_players and not config.reload_boxscores:
        for game_results in prior_data_games:
            game_obj = config.Game.get_game(game_results['id'])
            if game_obj and game_obj.game_date.date() < config.curr_date:
                game_obj.update_game(game_results)
        for player_results in prior_data_players:
            if datetime.strptime(player_results['game'][1], "%Y-%m-%d").date() < config.curr_date:
                player_obj = config.Player.create_player(player_results['player_id'])
                i = player_results['i']
                game = player_results['game']
                player_obj.update_player(game[0], game[1], game[3 - i], game[4], player_results)

    if (prior_data_games is None) or config.reload_boxscores:
        save_game_results = []
        save_player_results = []
        # Filter games to only include past games
        games = [game for game in config.Game.get_games()
                if game[1].date() < config.curr_date]

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            results = list(executor.map(lambda game: process_game(game, config), games))

        for game_result, player_result in results:
            if game_result:
                save_game_results.extend(game_result)
            if player_result:
                save_player_results.extend(player_result)

        config.save_data(dimension_games, save_game_results)
        config.save_data(dimension_players, save_player_results)


def process_goalie_data(data):
    results = dict()
    results['player_id'] = data['playerId']
    results['sweater_number'] = data['sweaterNumber']
    results['es_shots'] = data['evenStrengthShotsAgainst']
    results['pp_shots'] = data['powerPlayShotsAgainst']
    results['sh_shots'] = data['shorthandedShotsAgainst']
    results['save_shots'] = data['saveShotsAgainst']
    results['es_goals_against'] = data['evenStrengthGoalsAgainst']
    results['pp_goals_against'] = data['powerPlayGoalsAgainst']
    results['sh_goals_against'] = data['shorthandedGoalsAgainst']
    results['pim'] = None
    results['goals_against'] = data['goalsAgainst']
    results['toi'] = data['toi']
    results['starter'] = None

    return results


def process_forward_defense_data(data):
    results = dict()
    results['player_id'] = data['playerId']
    results['sweater_number'] = data['sweaterNumber']
    results['goals'] = data['goals']
    results['assists'] = data['assists']
    results['points'] = data['points']
    results['plus_minus'] = data['plusMinus']
    results['pim'] = data['pim']
    results['hits'] = data['hits']
    results['ppg'] = data['powerPlayGoals']
    results['shots'] = data['sog']
    results['faceoff_pctg'] = data['faceoffWinningPctg']
    results['toi'] = data['toi']

    return results


def process_boxscore(game_id, data_v1, data_v2):
    results = dict()
    results['id'] = game_id
    results['game_date'] = data_v1['gameDate']
    results['awayTeam'] = data_v1['awayTeam']['abbrev']
    results['homeTeam'] = data_v1['homeTeam']['abbrev']
    if data_v1['gameState'] != 'OFF':
        if data_v1['gameState'] == 'FUT':
            print(f'game_id: {game_id} {data_v1["awayTeam"]["name"]["default"]} vs {data_v1["homeTeam"]["name"]["default"]} postponed...')
            return False, results
        else:
            raise ValueError(f'game_state: {data_v1["gameState"]}')

    try:
        results['home_goals'] = data_v2['linescore']['totals']['home']
    except Exception as e:
        print(f'Exception: {e}')

    results['away_goals'] = data_v2['linescore']['totals']['away']
    results['playbyplay'] = data_v2['gameReports']['playByPlay']
    for category in data_v2['teamGameStats']:
        if category['category'] == 'sog':
            results['home_sog'] = category['homeValue']
            results['away_sog'] = category['awayValue']
        elif category['category'] == 'powerPlayPctg':
            results['home_pppctg'] = category['homeValue']
            results['away_pppctg'] = category['awayValue']
        elif category['category'] == 'giveaways':
            results['home_give'] = category['homeValue']
            results['away_give'] = category['awayValue']
        elif category['category'] == 'takeaways':
            results['home_take'] = category['homeValue']
            results['away_take'] = category['awayValue']
        elif category['category'] == 'hits':
            results['home_hits'] = category['homeValue']
            results['away_hits'] = category['awayValue']
        elif category['category'] == 'blockedShots':
            results['home_blocks'] = category['homeValue']
            results['away_blocks'] = category['awayValue']
        elif category['category'] == 'pim':
            results['home_pim'] = category['homeValue']
            results['away_pim'] = category['awayValue']
        elif category['category'] == 'faceoffWinningPctg':
            results['home_face'] = category['homeValue']
            results['away_face'] = category['awayValue']
        elif category['category'] == 'powerPlay':
            results['home_ppg'], results['home_powerplays'] = category['homeValue'].split('/')
            results['away_ppg'], results['away_powerplays'] = category['awayValue'].split('/')
            results['home_ppg'] = int(results['home_ppg'])
            results['home_powerplays'] = int(results['home_powerplays'])
            results['away_ppg'] = int(results['away_ppg'])
            results['away_powerplays'] = int(results['away_powerplays'])
            results['home_opp_powerplays'] = results['away_powerplays']
            results['home_pk_success'] = results['away_powerplays'] - results['away_ppg']
            results['away_opp_powerplays'] = results['home_powerplays']
            results['away_pk_success'] = results['home_powerplays'] - results['home_ppg']
    return True, results


def internal_process_game(game, config):
    """
    Process a single game:
      - Fetch the play-by-play page and process shifts.
      - Fetch the play data and process plays and game rosters.
    Returns a tuple: (results_shifts, results_plays, results_game_rosters)
    """
    print(f'Getting shift data for {game[0]}:{game[1]}:{game[2]}:{game[3]}')

    # Process shifts
    game_obj = config.Game.get_game(game[0])
    final_url = game_obj.playbyplay
    response = requests.get(final_url)
    response.raise_for_status()  # Ensure the response is OK
    soup = BeautifulSoup(response.text, 'html.parser')
    results_shifts = process_shifts(config, game, soup)

    # Process plays and game rosters
    final_url_plays = config.get_endpoint("plays", game_id=game[0])
    response_plays = requests.get(final_url_plays)
    response_plays.raise_for_status()  # Ensure the response is OK
    data_plays = response_plays.json()
    results_plays = process_plays(data_plays)
    results_game_rosters = process_game_rosters(data_plays, game_date=game[1])

    return results_shifts, results_plays, results_game_rosters


def get_playbyplay_data(config):
    print('Gathering play by play data...')
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"

    prior_data_shifts = config.load_data(dimension_shifts)
    prior_data_plays = config.load_data(dimension_plays)
    prior_data_game_rosters = config.load_data(dimension_game_rosters)

    if prior_data_shifts and config.reload_playbyplay is False:
        for results in prior_data_plays:
            pass  # Your existing code logic

    if (prior_data_shifts is None) or (config.reload_playbyplay is True):
        # Filter out future games before processing
        games = [game for game in config.Game.get_games()
                 if game[1].date() < config.curr_date]

        # Create a process pool
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            # Start the monitoring in a separate thread
            futures_dict = {}

            # Submit all jobs and track their start times
            for game in games:
                future = executor.submit(internal_process_game, game, config)
                futures_dict[future] = {
                    'start_time': time.time(),
                    'game': game
                }

            # Collect results as they complete
            save_results_shifts = []
            save_results_plays = []
            save_results_game_rosters = []

            try:
                for future in as_completed(futures_dict.keys()):
                    try:
                        result = future.result()
                        save_results_shifts.append(result[0])
                        save_results_plays.append(result[1])
                        save_results_game_rosters.append(result[2])
                    except Exception as exc:
                        game = futures_dict[future]['game']
                        print(f"Error processing game {game[0]} ({game[2]} vs {game[3]}): {exc}")
            finally:
                pass

        # Save all the collected data
        config.save_data(dimension_shifts, save_results_shifts)
        config.save_data(dimension_plays, save_results_plays)
        config.save_data(dimension_game_rosters, save_results_game_rosters)


# def get_playbyplay_data(config):
#     print('Gathering play by play data...')
#     dimension_shifts = "all_shifts"
#     dimension_plays = "all_plays"
#     dimension_game_rosters = "all_game_rosters"
#
#     prior_data_shifts = config.load_data(dimension_shifts)
#     prior_data_plays = config.load_data(dimension_plays)
#     prior_data_game_rosters = config.load_data(dimension_game_rosters)
#
#     if prior_data_shifts and config.reload_playbyplay is False:
#         for results in prior_data_plays:
#             pass  # Your existing code logic
#
#     if (prior_data_shifts is None) or (config.reload_playbyplay is True):
#         # Filter out future games before processing
#         games = [game for game in config.Game.get_games()
#                 if game[1].date() < config.curr_date]
#
#         with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
#             func = functools.partial(internal_process_game, config=config)
#             results = list(executor.map(func, games))
#
#         save_results_shifts = [result[0] for result in results]
#         save_results_plays = [result[1] for result in results]
#         save_results_game_rosters = [result[2] for result in results]
#
#         config.save_data(dimension_shifts, save_results_shifts)
#         config.save_data(dimension_plays, save_results_plays)
#         config.save_data(dimension_game_rosters, save_results_game_rosters)


def process_shifts(config, game, soup):
    print(f'Gathering player names for {game[0]}:{game[1]}:{game[2]}:{game[3]}')
    shifts = []
    pages = soup.find_all('div', class_='page')

    event_id = 1
    for i, page in enumerate(pages):

        target_tr = page.find('td', class_='tborder').parent

        # Get all subsequent <tr> tags
        subsequent_trs = target_tr.find_next_siblings('tr')

        for j, tr in enumerate(subsequent_trs):
            if j == 0:
                continue
            shift = dict()
            tds = tr.find_all('td')
            if check_copyright(tr.get_text(strip=True)):
                continue
            shift['event_id'] = event_id
            event_id = event_id + 1
            shift['period'] = tds[1].get_text(strip=True)
            shift['player_cnt'] = tds[2].get_text(strip=True)
            if tds[3].find('br'):
                # Split the content on the <br/> tag, preserving the formatting
                time_parts = tds[3].get_text(separator=' ').split()
            shift['elapsed_time'] = time_parts[0]
            shift['game_time'] = time_parts[1]
            shift['event_type'] = tds[4].get_text(strip=True)
            shift['desc'] = tds[5].get_text(strip=True)

            if len(tds) < 11 or (shift['event_type'] in ['PSTR', 'PGEND', 'GEND', 'ANTHEM']):
                shift['away_players'] = []
                shift['home_players'] = []
                shifts.append(shift)
                continue
            else:
                num_away_players = len(tds[6].find_all('table')) - 1
                away_players = []
                away_player_list = [7 + (4 * x) for x in range(0, num_away_players)]
                # away_player_cells = len(tds[6].find_all('td', class_='+ bborder + rborder'))

                for away_player in away_player_list:
                    sweater_number, position = split_data(tds[away_player].get_text(strip=True))
                    away_players.append((sweater_number, position))
                shift['away_players'] = away_players
                if not away_player_list:
                    home_tds_ind = 3
                else:
                    home_tds_ind = max(away_player_list) + 3
                num_home_players = len(tds[home_tds_ind].find_all('table')) - 1
                home_players = []
                home_player_list = [home_tds_ind + 1 + (4 * x) for x in range(0, num_home_players)]

                for home_player in home_player_list:
                    sweater_number, position = split_data(tds[home_player].get_text(strip=True))
                    home_players.append((sweater_number, position))
                shift['home_players'] = home_players
                shifts.append(shift)

    return shifts


def process_game_rosters(data, game_date):
    away_team = data['awayTeam']['abbrev']
    home_team = data['homeTeam']['abbrev']
    rosters = list()
    teams = dict()
    teams[data['awayTeam']['id']] = data['awayTeam']['abbrev']
    teams[data['homeTeam']['id']] = data['homeTeam']['abbrev']
    away_cnt = 0
    home_cnt = 0
    for player in data['rosterSpots']:
        if teams[player['teamId']] == away_team:
            away_cnt += 1
        else:
            home_cnt += 1
        roster = dict()
        roster['game_id'] = data['id']
        roster['game_date'] = game_date
        roster['player_team'] = teams[player['teamId']]
        roster['player_id'] = player['playerId']
        roster['player_first_name'] = player['firstName']
        roster['player_last_name'] = player['lastName']
        roster['player_sweater'] = player['sweaterNumber']
        roster['player_position'] = player['positionCode']
        rosters.append(roster)

    default_id = 1000000
    # Pad away team roster if less than 20
    if away_cnt < 20:
        missing = 20 - away_cnt
        for _ in range(missing):
            dummy_player = create_dummy_player(
                game_id=data['id'],
                team_abbrev=away_team,
                game_date=game_date,
                player_id= default_id,
            )
            default_id += 1
            print(f'padding players for {data["id"]}')
            rosters.append(dummy_player)

    # Pad home team roster if less than 20
    if home_cnt < 20:
        missing = 20 - home_cnt
        for _ in range(missing):
            dummy_player = create_dummy_player(
                game_id=data['id'],
                team_abbrev=home_team ,
                game_date=game_date,
                player_id=default_id,
            )
            default_id += 1
            print(f'padding players for {data["id"]}')
            rosters.append(dummy_player)

    return rosters

def process_plays(data):
    plays = list()
    for play in data['plays']:
        shift = dict()
        shift['game_id'] = data['id']
        shift['event_id'] = play['eventId']
        shift['period'] = play['periodDescriptor']['number']
        shift['period_type'] = play['periodDescriptor']['periodType']
        shift['overtime'] = False
        shift['shootout'] = False
        if shift['period_type'] == 'OT':
            shift['overtime'] = True
        elif shift['period_type'] == 'SO':
            shift['shootout'] = True
        shift['elapsed_time'] = play['timeInPeriod']
        shift['game_time'] = play['timeRemaining']
        shift['event_type'] = play['typeDescKey']
        shift['event_code'] = play['typeCode']
        shift['stoppage'] = None
        shift['faceoff_winner'] = None
        shift['faceoff_loser'] = None
        shift['hitting_player'] = None
        shift['hittee_player'] = None
        shift['giveaway'] = None
        shift['takeaway'] = None
        shift['goal'] = None
        shift['shot_attempt'] = None
        shift['shot_on_goal'] = None
        shift['goal_assist1'] = None
        shift['goal_assist2'] = None
        shift['goal_against'] = None
        shift['shot_saved'] = None
        shift['missed_shot_attempt'] = None
        shift['blocked_shot_attempt'] = None
        shift['blocked_shot_saved'] = None
        shift['icing'] = None
        shift['penalty_committed'] = None
        shift['penalty_served'] = None
        shift['penalty_drawn'] = None
        shift['penalty_duration'] = None
        shift['delayed_penalty'] = None
        shift['penalty_match'] = None
        shift['penalty_shot'] = None
        shift['penalty_shot_goal'] = None
        shift['penalty_shot_saved'] = None
        shift['period_end'] = None
        shift['game_end'] = None
        if play.get('typeCode') == 502:
            shift['faceoff_winner'] = play['details'].get('winningPlayerId', None)
            shift['faceoff_loser'] = play['details'].get('losingPlayerId', None)
        elif play.get('typeCode') == 503:
            shift['hitting_player'] = play['details'].get('hittingPlayerId', None)
            shift['hittee_player'] = play['details'].get('hitteePlayerId', None)
        elif play.get('typeCode') == 504:
            shift['giveaway'] = play['details'].get('playerId', None)
        elif play.get('typeCode') == 505:
            shift['goal'] = play['details'].get('scoringPlayerId', None)
            shift['shot_attempt'] = play['details'].get('scoringPlayerId', None)
            shift['shot_on_goal'] = play['details'].get('scoringPlayerId', None)
            shift['goal_assist1'] = play['details'].get('assist1PlayerId', None)
            shift['goal_assist2'] = play['details'].get('assist2PlayerId', None)
            shift['goal_against'] = play['details'].get('goalieInNetId', None)
        elif play.get('typeCode') == 506:
            shift['shot_attempt'] = play['details'].get('shootingPlayerId', None)
            shift['shot_on_goal'] = play['details'].get('shootingPlayerId', None)
            shift['shot_saved'] = play['details'].get('goalieInNetId', None)
        elif play.get('typeCode') == 507:
            shift['missed_shot_attempt'] = play['details'].get('shootingPlayerId', None)
        elif play.get('typeCode') == 508:
            shift['blocked_shot_attempt'] = play['details'].get('shootingPlayerId', None)
            shift['blocked_shot_saved'] = play['details'].get('blockingPlayerId', None)
        elif play.get('typeCode') == 509:
            if play['details'].get('duration', None) is not None:
                if play['details'].get('typeCode', None) == 'MAT':
                    shift['penalty_duration'] = play['details'].get('duration', None) - 10
                else:
                    shift['penalty_duration'] = play['details'].get('duration', None)
            if play['details'].get('committedByPlayerId', None) is not None:
                shift['penalty_committed'] = play['details'].get('committedByPlayerId', None)
                shift['penalty_drawn'] = play['details'].get('drawnByPlayerId', None)
                shift['penalty_served']= play['details'].get('servedByPlayerId', None)
            elif play['details'].get('servedByPlayerId', None) is not None:
                shift['penalty_served'] = play['details'].get('servedByPlayerId', None)
        elif play.get('typeCode') == 510:
            cwc = 0
        elif play.get('typeCode') == 516:
            shift['stoppage'] = play['details'].get('reason', None)
            if shift['stoppage'] not in ['icing', 'goalie-stopped-after-sog', 'puck-in-crowd', 'puck-in-netting',
                                 'offside', 'puck-in-crowd', 'puck-in-benches', 'puck-frozen', 'tv-timeout',
                                 'high-stick', 'net-dislodged-defensive-skater', 'player-injury', 'video-review',
                                 'referee-or-linesman', 'clock-problem',
                                 'hand-pass', 'objects-on-ice', 'goalie-puck-frozen-played-from-beyond-center',
                                 'visitor-timeout', 'net-dislodged-offensive-skater', 'chlg-hm-goal-interference',
                                 'chlg-vis-goal-interference', 'chlg-hm-missed-stoppage',
                                 'skater-puck-frozen', 'ice-scrape', 'chlg-league-goal-interference',
                                 'player-equipment', 'chlg-hm-off-side', 'chlg-vis-off-side',
                                 'chlg-hm-missed-stoppage', 'home-timeout', 'chlg-vis-missed-stoppage',
                                 'puck-in-penalty-benches', 'ice-problem', 'net-dislodged-by-goaltender',
                                 'rink-repair', 'chlg-league-missed-stoppage', 'official-injury', 'chlg-hm-puck-over-glass',
                                 'chlg-league-off-side','switch-sides']:
                print(f'\n')
                print(f'collect play stoppage reason: {play["details"]["reason"]}')
                print(f'\n')
        elif play.get('typeCode') in [520, 524]:  # period-start / game-end
            pass
        elif play.get('typeCode') == 521: # period-end
            shift['period_end'] = True
        elif play.get('typeCode') == 523: # period-end
            shift['period_end'] = True
        elif play.get('typeCode') == 525:
            shift['takeaway'] = play['details'].get('playerId', None)
        elif play.get('typeCode') == 535: # delayed-penalty
            shift['delayed_penalty'] = 1
        elif play.get('typeCode') == 537: # penalty-shot-missed
            shift['penalty_shot'] = play['details'].get('shootingPlayerId', None)
            shift['penalty_shot_saved'] = play['details'].get('goalieInNetId', None)
        else:
            cwc = 0

        plays.append(shift)

    return plays


# def get_player_names(config):
#     print(f'Gathering player names...')
#     dimension = "all_names"
#     prior_data = config.load_data(dimension)
#     if prior_data and config.reload_playernames is False:
#         for player_data in prior_data:
#             player_id = player_data['playerId']
#             last_name = player_data['lastName']['default']
#             first_name = player_data['firstName']['default']
#             player_obj = config.Player.get_player(player_id)
#             player_obj.update_name(last_name, first_name)
#
#             # update player names
#     if (prior_data is None) or (config.reload_playernames is True):
#         save_data = []
#         for i, player_id in enumerate(config.Player.get_players()):
#             if i % 100 == 0:
#                 print(f'{i} players gathered..')
#             final_url = config.get_endpoint("player", player_id=player_id)
#             response = requests.get(final_url)
#             response.raise_for_status()  # Ensure the response is OK
#             player_data = response.json()
#
#             last_name = player_data['lastName']['default']
#             first_name = player_data['firstName']['default']
#             player_obj = config.Player.get_player(player_id)
#             player_obj.update_name(last_name, first_name)
#             save_data.append(player_data)
#
#         # Set the games in the config, assuming this modifies some shared state or configuration
#         config.save_data(dimension, save_data)


def get_player_names(config):
    print("Gathering player names...")
    dimension = "all_names"
    prior_data = config.load_data(dimension)

    # If we already have data and do not need to reload, update from the stored data.
    if prior_data and not config.reload_playernames:
        for player_data in prior_data:
            player_id = player_data['playerId']
            last_name = player_data['lastName']['default']
            first_name = player_data['firstName']['default']
            player_obj = config.Player.get_player(player_id)
            player_obj.update_name(last_name, first_name)

    # Otherwise, fetch the data via HTTP calls in parallel.
    if (prior_data is None) or (config.reload_playernames is True):
        # Get the list of player IDs in the same order as the serial code.
        players = list(config.Player.get_players())

        # Define a helper function that does the network call, processes the data,
        # and returns the player_data. This function will be executed concurrently.
        def fetch_and_update(player_id):
            final_url = config.get_endpoint("player", player_id=player_id)
            response = requests.get(final_url)
            response.raise_for_status()  # Make sure the request was successful.
            player_data = response.json()

            # Update the player's name using the fetched data.
            last_name = player_data['lastName']['default']
            first_name = player_data['firstName']['default']
            player_obj = config.Player.get_player(player_id)
            player_obj.update_name(last_name, first_name)

            return player_data

        save_data = []
        # You can adjust max_workers as needed. Since these calls are I/O-bound,
        # using threads is appropriate.
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # executor.map ensures that results are yielded in the same order as the input.
            for i, result in enumerate(executor.map(fetch_and_update, players)):
                if i % 100 == 0:
                    print(f"{i} players gathered..")
                save_data.append(result)

        # Save the collected data (in order) using config's method.
        config.save_data(dimension, save_data)


def split_data(combined_component):
    match = re.match(r'(\d+)([A-Za-z]+)', combined_component)
    if match:
        numeric_part = match.group(1)
        text_part = match.group(2)
    else:
        numeric_part = ''
        text_part = ''
    return numeric_part, text_part


def check_copyright(text):
    # Check if the copyright symbol and "Copyright" are present
    if "©" not in text or "Copyright" not in text:
        return False

    # Check if the year is in the expected range
    check = False
    for yr in ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
        if yr in text:
            check = True

    if check:
        return True
    else:
        return False