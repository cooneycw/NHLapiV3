from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src_code.utils.utils import create_dummy_player
import os
import re
import requests


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


def get_boxscore_list(config):
    print('Gathering boxscore data...')
    dimension_games = "all_boxscores"
    dimension_players = "all_players"
    prior_data_games = config.load_data(dimension_games)
    prior_data_players = config.load_data(dimension_players)

    # Get the list of selected seasons based on config.season_count
    selected_seasons = [season[0] for season in config.Season.get_selected_seasons(config.season_count)]
    print(
        f'Processing boxscore data for {len(selected_seasons)} seasons: {", ".join(str(s) for s in selected_seasons)}')

    # If we have valid cached data and don't need a full reload
    if prior_data_games and prior_data_players and not config.reload_boxscores:
        print(
            f'Using cached boxscore data ({len(prior_data_games)} games, {len(prior_data_players)} player entries)...')

        # Create a set of game IDs that we already have boxscore data for
        cached_game_ids = {game_results['id'] for game_results in prior_data_games}

        # Load the cached data into memory (only for selected seasons)
        print("Loading cached game data into memory...")
        loaded_game_count = 0
        for game_results in prior_data_games:
            game_obj = config.Game.get_game(game_results['id'])
            # Only process games from selected seasons - ensure both are same type for comparison
            if (game_obj and game_obj.game_date.date() < config.curr_date and
                    str(game_obj.season_id) in [str(s) for s in selected_seasons]):  # Changed from season to season_id
                game_obj.update_game(game_results)
                loaded_game_count += 1
        print(f"Loaded {loaded_game_count} games from selected seasons")

        print("Loading cached player data into memory...")
        player_count = 0
        for player_results in prior_data_players:
            # Handle the datetime object properly - check if it's already a datetime or needs parsing
            game_date = player_results['game'][1]
            if isinstance(game_date, datetime):
                game_date = game_date.date()
            else:
                game_date = datetime.strptime(game_date, "%Y-%m-%d").date()

            if game_date < config.curr_date:
                # Check if this player's game is from a selected season - ensure same type comparison
                if str(player_results['game'][4]) in [str(s) for s in selected_seasons]:
                    try:
                        player_obj = config.Player.create_player(player_results['player_id'])
                        i = player_results['i']
                        game = player_results['game']
                        player_obj.update_player(game[0], game[1], game[3 - i], game[4], player_results)
                        player_count += 1
                        if player_count % 1000 == 0:
                            print(f"Loaded {player_count} player game records...")
                    except Exception as e:
                        print(f"Error loading player data: {e}")
        print(f"Loaded {player_count} player game records from selected seasons")

        # Rest of the function remains the same...
        # Find all completed games from selected seasons that aren't in our cache yet
        # Convert to strings for comparison
        str_selected_seasons = [str(s) for s in selected_seasons]
        all_completed_games = [game for game in config.Game.get_games()
                               if game[1].date() < config.curr_date and str(game[4]) in str_selected_seasons]
        games_needing_boxscores = [game for game in all_completed_games
                                   if game[0] not in cached_game_ids]

        if games_needing_boxscores:
            print(f'Found {len(games_needing_boxscores)} new games that need boxscore data...')

            # Fetch boxscores only for the new games
            # For filtering cached data, ensure type-safe comparisons
            save_game_results = [g for g in prior_data_games
                                 if g['id'] in cached_game_ids and
                                 str(g.get('season_id', g.get('game_date', '').split('-')[
                                     0])) in str_selected_seasons]  # Changed from season to season_id

            save_player_results = [p for p in prior_data_players
                                   if str(p['game'][4]) in str_selected_seasons]

            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Use a list to collect results safely
                futures = [executor.submit(process_game, game, config) for game in games_needing_boxscores]

                completed = 0
                for future in as_completed(futures):
                    try:
                        game_result, player_result = future.result()
                        if game_result:
                            save_game_results.extend(game_result)
                        if player_result:
                            save_player_results.extend(player_result)

                        completed += 1
                        if completed % 10 == 0 or completed == len(games_needing_boxscores):
                            print(f"Processed {completed}/{len(games_needing_boxscores)} games...")
                    except Exception as e:
                        print(f"Error processing game in executor: {e}")

            # Save the updated data
            print(
                f'Saving updated boxscore data ({len(save_game_results)} games, {len(save_player_results)} player entries)...')
            config.save_data(dimension_games, save_game_results)
            config.save_data(dimension_players, save_player_results)
        else:
            print('No new games need boxscore data, using cached data only.')
    else:
        # We need to fetch all boxscore data (either no cache or full reload requested)
        reason = "Reload requested." if config.reload_boxscores else "No valid cached data."
        print(f'{reason} Fetching all boxscore data for selected seasons...')

        save_game_results = []
        save_player_results = []

        # Filter games to only include past games from selected seasons
        str_selected_seasons = [str(s) for s in selected_seasons]
        games = [game for game in config.Game.get_games()
                 if game[1].date() < config.curr_date and str(game[4]) in str_selected_seasons]

        print(f'Fetching boxscore data for {len(games)} completed games from selected seasons...')

        # Use as_completed for better error handling
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [executor.submit(process_game, game, config) for game in games]

            completed = 0
            for future in as_completed(futures):
                try:
                    game_result, player_result = future.result()
                    if game_result:
                        save_game_results.extend(game_result)
                    if player_result:
                        save_player_results.extend(player_result)

                    completed += 1
                    if completed % 10 == 0 or completed == len(games):
                        print(f"Processed {completed}/{len(games)} games...")
                except Exception as e:
                    print(f"Error processing game in executor: {e}")

        print(f'Saving boxscore data ({len(save_game_results)} games, {len(save_player_results)} player entries)...')
        config.save_data(dimension_games, save_game_results)
        config.save_data(dimension_players, save_player_results)


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


def get_playbyplay_data(config):
    print('Gathering play by play data...')
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"

    # Handle selective file deletion if requested
    if config.delete_files:
        for dimension in [dimension_shifts, dimension_plays, dimension_game_rosters]:
            file_path = config.file_paths[dimension]
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")

    # Get the list of selected seasons based on config.season_count
    selected_seasons = [season[0] for season in config.Season.get_selected_seasons(config.season_count)]
    str_selected_seasons = [str(s) for s in selected_seasons]
    print(
        f'Processing play-by-play data for {len(selected_seasons)} seasons: {", ".join(str(s) for s in selected_seasons)}')

    # Load existing data
    prior_data_shifts = config.load_data(dimension_shifts)
    prior_data_plays = config.load_data(dimension_plays)
    prior_data_game_rosters = config.load_data(dimension_game_rosters)

    # Check if we have valid data and don't need a full reload
    if (prior_data_shifts and prior_data_plays and prior_data_game_rosters and
            not config.reload_playbyplay):
        print(f'Using cached play-by-play data:')
        print(f' - {len(prior_data_shifts)} shift records')
        print(f' - {len(prior_data_plays)} play records')
        print(f' - {len(prior_data_game_rosters)} game roster records')

        # Create dictionaries for existing data for quick lookups
        existing_game_ids_shifts = set()
        for shift_data in prior_data_shifts:
            # Extract game IDs from shift data (assuming each shift has game info)
            if shift_data and len(shift_data) > 0:
                for shift in shift_data:
                    if 'game_id' in shift:
                        existing_game_ids_shifts.add(shift['game_id'])
                    break  # Only need to check the first shift to get game ID

        existing_game_ids_plays = set(play_data['game_id'] for play_data in prior_data_plays if 'game_id' in play_data)
        existing_game_ids_rosters = set()
        for roster_list in prior_data_game_rosters:
            if roster_list and len(roster_list) > 0:
                existing_game_ids_rosters.add(roster_list[0]['game_id'])

        # Get the list of all completed games in the selected seasons
        all_completed_games = [game for game in config.Game.get_games()
                               if game[1].date() < config.curr_date and
                               str(game[4]) in str_selected_seasons]  # Filter by selected seasons

        print(f'Found {len(all_completed_games)} completed games in selected seasons')

        # Identify games that need playbyplay data
        games_needing_playbyplay = []
        for game in all_completed_games:
            game_id = game[0]
            # Check if this game needs any of the data types
            if (game_id not in existing_game_ids_shifts or
                    game_id not in existing_game_ids_plays or
                    game_id not in existing_game_ids_rosters):
                games_needing_playbyplay.append(game)

        # Also check for roster updates for today's games if roster reload is enabled
        if config.reload_rosters:
            todays_games = [game for game in config.Game.get_games()
                            if game[1].date() == config.curr_date and
                            str(game[4]) in str_selected_seasons]  # Filter by selected seasons

            # Add today's games to the processing list for roster data
            for game in todays_games:
                if game not in games_needing_playbyplay:
                    games_needing_playbyplay.append(game)
                    print(f"Adding today's game {game[0]} ({game[2]} @ {game[3]}) for roster update")

        if games_needing_playbyplay:
            print(f'Found {len(games_needing_playbyplay)} games that need play-by-play data...')

            # Process only the games that need updates
            new_shifts_data = []
            new_plays_data = []
            new_rosters_data = []

            # Batch the games to optimize memory usage
            batch_size = min(100, len(games_needing_playbyplay))  # Adjust batch size as needed
            batches = [games_needing_playbyplay[i:i + batch_size] for i in
                       range(0, len(games_needing_playbyplay), batch_size)]

            # Process each batch
            completed_count = 0
            total_count = len(games_needing_playbyplay)

            for batch_idx, batch in enumerate(batches):
                print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} games)...")

                # Use ThreadPoolExecutor for I/O-bound operations
                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    # Submit batch of jobs
                    future_to_game = {executor.submit(internal_process_game, game, config): game for game in batch}

                    # Process results as they complete
                    for future in as_completed(future_to_game):
                        try:
                            game = future_to_game[future]
                            result = future.result()

                            # Unpack results
                            shifts_result, plays_result, game_rosters_result = result

                            # Add results to our data lists
                            if shifts_result:
                                new_shifts_data.append(shifts_result)
                            if plays_result:
                                new_plays_data.append(plays_result)
                            if game_rosters_result:
                                new_rosters_data.append(game_rosters_result)

                            completed_count += 1
                            if completed_count % 10 == 0 or completed_count == total_count:
                                print(f"Processed {completed_count}/{total_count} games...")
                        except Exception as exc:
                            game = future_to_game[future]
                            print(f"Error processing game {game[0]} ({game[2]} @ {game[3]}): {exc}")
                            import traceback
                            traceback.print_exc()

            # Filter existing data by selected seasons
            # For shifts data
            filtered_shifts_data = []
            for shift_data in prior_data_shifts:
                if shift_data and len(shift_data) > 0:
                    game_id = None
                    for shift in shift_data:
                        if 'game_id' in shift:
                            game_id = shift['game_id']
                            break

                    # Check if this game is in our selected seasons
                    # We need to look up the game to find its season
                    if game_id:
                        game_obj = config.Game.get_game(game_id)
                        if game_obj and str(game_obj.season_id) in str_selected_seasons:
                            # Only keep data for selected seasons
                            filtered_shifts_data.append(shift_data)

            # For plays data
            filtered_plays_data = []
            for play_data in prior_data_plays:
                if 'game_id' in play_data:
                    game_id = play_data['game_id']
                    game_obj = config.Game.get_game(game_id)
                    if game_obj and str(game_obj.season_id) in str_selected_seasons:
                        filtered_plays_data.append(play_data)

            # For roster data
            filtered_rosters_data = []
            for roster_list in prior_data_game_rosters:
                if roster_list and len(roster_list) > 0:
                    game_id = roster_list[0]['game_id']
                    game_obj = config.Game.get_game(game_id)
                    if game_obj and str(game_obj.season_id) in str_selected_seasons:
                        filtered_rosters_data.append(roster_list)

            # Merge new data with existing data, handling conflicts
            # For shifts: replace existing game data with new data
            merged_shifts_data = []
            game_ids_to_update = [g[0] for g in games_needing_playbyplay]

            # Add existing shifts data, excluding those we're updating
            for shift_data in filtered_shifts_data:
                if shift_data and len(shift_data) > 0:
                    game_id = None
                    for shift in shift_data:
                        if 'game_id' in shift:
                            game_id = shift['game_id']
                            break

                    if game_id and game_id not in game_ids_to_update:
                        merged_shifts_data.append(shift_data)

            # Add new shifts data
            merged_shifts_data.extend(new_shifts_data)

            # For plays: similar approach
            merged_plays_data = []
            for play_data in filtered_plays_data:
                if 'game_id' in play_data and play_data['game_id'] not in game_ids_to_update:
                    merged_plays_data.append(play_data)
            merged_plays_data.extend(new_plays_data)

            # For game rosters: similar approach
            merged_rosters_data = []
            for roster_list in filtered_rosters_data:
                if roster_list and len(roster_list) > 0 and roster_list[0]['game_id'] not in game_ids_to_update:
                    merged_rosters_data.append(roster_list)
            merged_rosters_data.extend(new_rosters_data)

            # Save the merged data
            print(f'Saving updated play-by-play data:')
            print(f' - {len(merged_shifts_data)} shift records')
            print(f' - {len(merged_plays_data)} play records')
            print(f' - {len(merged_rosters_data)} game roster records')

            config.save_data(dimension_shifts, merged_shifts_data)
            config.save_data(dimension_plays, merged_plays_data)
            config.save_data(dimension_game_rosters, merged_rosters_data)
        else:
            print('No games need play-by-play data updates for the selected seasons')
    else:
        # We need to fetch all play-by-play data
        reason = "Reload requested." if config.reload_playbyplay else "No valid cached data."
        print(f'{reason} Fetching all play-by-play data for selected seasons...')

        # Filter games to only include past games from selected seasons
        games = [game for game in config.Game.get_games()
                 if game[1].date() < config.curr_date and
                 str(game[4]) in str_selected_seasons]  # Filter by selected seasons

        # If roster reload is enabled, include today's games for roster data
        if config.reload_rosters:
            todays_games = [game for game in config.Game.get_games()
                            if game[1].date() == config.curr_date and
                            str(game[4]) in str_selected_seasons]  # Filter by selected seasons

            for game in todays_games:
                if game not in games:
                    games.append(game)
                    print(f"Including today's game {game[0]} ({game[2]} @ {game[3]}) for roster data")

        print(f'Processing {len(games)} games for play-by-play data...')

        # Batch the games to optimize memory usage
        batch_size = min(100, len(games))  # Adjust batch size as needed
        batches = [games[i:i + batch_size] for i in range(0, len(games), batch_size)]

        # Collect results
        save_results_shifts = []
        save_results_plays = []
        save_results_game_rosters = []

        completed_count = 0
        total_count = len(games)

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} games)...")

            # Use ThreadPoolExecutor for I/O-bound operations
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Submit batch of jobs
                future_to_game = {executor.submit(internal_process_game, game, config): game for game in batch}

                # Process results as they complete
                for future in as_completed(future_to_game):
                    try:
                        game = future_to_game[future]
                        result = future.result()

                        # Unpack results
                        shifts_result, plays_result, game_rosters_result = result

                        # Add results to our data lists
                        if shifts_result:
                            save_results_shifts.append(shifts_result)
                        if plays_result:
                            save_results_plays.append(plays_result)
                        if game_rosters_result:
                            save_results_game_rosters.append(game_rosters_result)

                        completed_count += 1
                        if completed_count % 10 == 0 or completed_count == total_count:
                            print(f"Processed {completed_count}/{total_count} games...")
                    except Exception as exc:
                        game = future_to_game[future]
                        print(f"Error processing game {game[0]} ({game[2]} @ {game[3]}): {exc}")
                        import traceback
                        traceback.print_exc()

            # Optionally save after each batch to prevent data loss in case of failure
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(batches):
                print(f'Saving intermediate results after batch {batch_idx + 1}...')
                print(f' - {len(save_results_shifts)} shift records')
                print(f' - {len(save_results_plays)} play records')
                print(f' - {len(save_results_game_rosters)} game roster records')

                config.save_data(dimension_shifts, save_results_shifts)
                config.save_data(dimension_plays, save_results_plays)
                config.save_data(dimension_game_rosters, save_results_game_rosters)

        # Final save
        print(f'Saving final play-by-play data:')
        print(f' - {len(save_results_shifts)} shift records')
        print(f' - {len(save_results_plays)} play records')
        print(f' - {len(save_results_game_rosters)} game roster records')

        config.save_data(dimension_shifts, save_results_shifts)
        config.save_data(dimension_plays, save_results_plays)
        config.save_data(dimension_game_rosters, save_results_game_rosters)


def get_player_names(config):
    print("Gathering player names...")
    dimension = "all_names"
    prior_data = config.load_data(dimension)

    # Get the list of selected seasons
    selected_seasons = [season[0] for season in config.Season.get_selected_seasons(config.season_count)]
    str_selected_seasons = [str(s) for s in selected_seasons]
    print(f'Processing player names for {len(selected_seasons)} seasons: {", ".join(str(s) for s in selected_seasons)}')

    # If we already have data and do not need to reload, update from the stored data.
    if prior_data and not config.reload_playernames:
        print(f'Using cached player name data ({len(prior_data)} players)...')
        cached_player_ids = set()

        # Load existing player names into memory
        for player_data in prior_data:
            player_id = player_data['playerId']
            cached_player_ids.add(player_id)
            last_name = player_data['lastName']['default']
            first_name = player_data['firstName']['default']
            player_obj = config.Player.get_player(player_id)
            player_obj.update_name(last_name, first_name)

        # Identify new players from our selected seasons who aren't in the cache
        all_players = set()
        for player_id in config.Player.get_players():
            player_obj = config.Player.get_player(player_id)
            # Check if this player played in any of our selected seasons
            player_seasons = [str(s) for s in player_obj.seasons]
            if any(season in str_selected_seasons for season in player_seasons):
                all_players.add(player_id)

        new_players = all_players - cached_player_ids

        if new_players:
            print(f'Found {len(new_players)} new players that need name data...')
            new_player_data = []

            # Use ThreadPoolExecutor for parallel requests (better for I/O bound operations)
            # Also batch the players to control memory usage
            batch_size = min(200, len(new_players))  # Adjust batch size as needed
            player_batches = [list(new_players)[i:i + batch_size] for i in range(0, len(new_players), batch_size)]

            for batch_idx, player_batch in enumerate(player_batches):
                print(f"Processing player batch {batch_idx + 1}/{len(player_batches)} ({len(player_batch)} players)...")

                with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                    # Define a function to fetch and process a single player
                    def fetch_player_name(player_id):
                        try:
                            final_url = config.get_endpoint("player", player_id=player_id)
                            response = requests.get(final_url)
                            response.raise_for_status()
                            player_data = response.json()

                            # Update the player's name
                            last_name = player_data['lastName']['default']
                            first_name = player_data['firstName']['default']
                            player_obj = config.Player.get_player(player_id)
                            player_obj.update_name(last_name, first_name)

                            return player_data
                        except Exception as e:
                            print(f"Error fetching player {player_id}: {e}")
                            return None

                    # Map is more efficient for simple operations with uniform processing time
                    for result in executor.map(fetch_player_name, player_batch):
                        if result:
                            new_player_data.append(result)

                # Print progress
                print(
                    f"Completed batch {batch_idx + 1}/{len(player_batches)}, processed {len(new_player_data)} players so far")

            # Merge with existing data and save
            save_data = list(prior_data) + new_player_data
            print(f'Saving updated player name data ({len(save_data)} players)...')
            config.save_data(dimension, save_data)
        else:
            print('No new players need name data, using cached data only.')
    else:
        # We need to fetch all player names (either no cache or full reload requested)
        reason = "Reload requested." if config.reload_playernames else "No valid cached data."
        print(f'{reason} Fetching all player names...')

        # Get the list of player IDs - filter by selected seasons
        players_to_fetch = []
        for player_id in config.Player.get_players():
            player_obj = config.Player.get_player(player_id)
            # Check if this player played in any of our selected seasons
            player_seasons = [str(s) for s in player_obj.seasons]
            if any(season in str_selected_seasons for season in player_seasons):
                players_to_fetch.append(player_id)

        print(f'Fetching names for {len(players_to_fetch)} players from selected seasons...')
        save_data = []

        # Batch the players to control memory usage
        batch_size = min(200, len(players_to_fetch))  # Adjust batch size as needed
        player_batches = [players_to_fetch[i:i + batch_size] for i in range(0, len(players_to_fetch), batch_size)]

        for batch_idx, player_batch in enumerate(player_batches):
            print(f"Processing player batch {batch_idx + 1}/{len(player_batches)} ({len(player_batch)} players)...")

            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Define a function to fetch and process a single player
                def fetch_player_name(player_id):
                    try:
                        final_url = config.get_endpoint("player", player_id=player_id)
                        response = requests.get(final_url)
                        response.raise_for_status()
                        player_data = response.json()

                        # Update the player's name
                        last_name = player_data['lastName']['default']
                        first_name = player_data['firstName']['default']
                        player_obj = config.Player.get_player(player_id)
                        player_obj.update_name(last_name, first_name)

                        return player_data
                    except Exception as e:
                        print(f"Error fetching player {player_id}: {e}")
                        return None

                # Map is more efficient for simple operations with uniform processing time
                for result in executor.map(fetch_player_name, player_batch):
                    if result:
                        save_data.append(result)

            # Intermediate save to prevent data loss
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(player_batches):
                print(
                    f'Saving intermediate player data after batch {batch_idx + 1} ({len(save_data)} players so far)...')
                config.save_data(dimension, save_data)

        print(f'Saving final player name data ({len(save_data)} players)...')
        config.save_data(dimension, save_data)




def internal_process_game(game, config):
    """
    Process a single game:
      - Fetch the play-by-play page and process shifts.
      - Fetch the play data and process plays and game rosters.

    For games that haven't been played yet (today's games), only fetch roster data.

    Returns a tuple: (results_shifts, results_plays, results_game_rosters)
    """
    game_id = game[0]
    game_date = game[1]
    away_team = game[2]
    home_team = game[3]

    # Handle datetime objects correctly
    if isinstance(game_date, datetime):
        game_date_obj = game_date.date()
    else:
        game_date_obj = datetime.strptime(game_date, "%Y-%m-%d").date()

    # If the game is today or in the future, only try to get roster data
    if game_date_obj >= config.curr_date:
        print(f'Getting roster data only for upcoming game {game_id}: {away_team} @ {home_team}')

        try:
            # Process game rosters only
            final_url_plays = config.get_endpoint("plays", game_id=game_id)
            response_plays = requests.get(final_url_plays)

            # Check if the response is successful
            if response_plays.status_code == 200:
                data_plays = response_plays.json()
                # Process rosters if available
                results_game_rosters = process_game_rosters(data_plays, game_date=game_date)

                # Maintain structure of return tuple but only return roster data
                return None, None, results_game_rosters
            else:
                print(f"Could not retrieve play data for game {game_id}: Status code {response_plays.status_code}")
                return None, None, None
        except Exception as e:
            print(f"Error retrieving roster data for game {game_id}: {str(e)}")
            return None, None, None

    # For past games, process full play-by-play data
    print(f'Getting complete play-by-play data for {game_id}: {away_team} @ {home_team}')

    # Get the game object to access playbyplay URL
    game_obj = config.Game.get_game(game_id)

    # Initialize results
    results_shifts = None
    results_plays = None
    results_game_rosters = None

    # Process shifts if playbyplay URL is available
    if hasattr(game_obj, 'playbyplay') and game_obj.playbyplay:
        try:
            final_url = game_obj.playbyplay
            response = requests.get(final_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results_shifts = process_shifts(config, game, soup)
            else:
                print(f"Could not retrieve shift data for game {game_id}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error processing shifts for game {game_id}: {str(e)}")

    # Process plays and game rosters
    try:
        final_url_plays = config.get_endpoint("plays", game_id=game_id)
        response_plays = requests.get(final_url_plays)

        if response_plays.status_code == 200:
            data_plays = response_plays.json()
            results_plays = process_plays(data_plays)
            results_game_rosters = process_game_rosters(data_plays, game_date=game_date)
        else:
            print(f"Could not retrieve play data for game {game_id}: Status code {response_plays.status_code}")
    except Exception as e:
        print(f"Error processing plays/rosters for game {game_id}: {str(e)}")

    return results_shifts, results_plays, results_game_rosters


def process_game_rosters(data, game_date):
    """
    Process roster data from the plays API endpoint.
    Handles creating roster entries for each player, padding with dummy players if needed.
    """
    # Ensure game_date is handled correctly - could be a datetime object or string
    if isinstance(game_date, datetime):
        game_date_obj = game_date
    else:
        try:
            game_date_obj = datetime.strptime(game_date, "%Y-%m-%d")
        except (TypeError, ValueError):
            # If there's an error, use current date as fallback
            print(f"Warning: Invalid game_date format: {game_date}. Using current date.")
            game_date_obj = datetime.now()

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
        roster['game_date'] = game_date_obj
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
                game_date=game_date_obj,  # Use the properly handled datetime object
                player_id=default_id,
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
                team_abbrev=home_team,
                game_date=game_date_obj,  # Use the properly handled datetime object
                player_id=default_id,
            )
            default_id += 1
            print(f'padding players for {data["id"]}')
            rosters.append(dummy_player)

    return rosters


def process_shifts(config, game, soup):
    """
    Process shift data from the HTML play-by-play page.
    Returns a list of shift events with player information.
    """
    print(f'Gathering player names for {game[0]}:{game[1]}:{game[2]}:{game[3]}')
    shifts = []
    pages = soup.find_all('div', class_='page')

    event_id = 1
    for i, page in enumerate(pages):
        target_tr = page.find('td', class_='tborder').parent
        if not target_tr:
            continue

        # Get all subsequent <tr> tags
        subsequent_trs = target_tr.find_next_siblings('tr')

        for j, tr in enumerate(subsequent_trs):
            if j == 0:
                continue
            shift = dict()
            tds = tr.find_all('td')
            if not tds or check_copyright(tr.get_text(strip=True)):
                continue

            shift['game_id'] = game[0]  # Add game_id to each shift
            shift['event_id'] = event_id
            event_id = event_id + 1

            if len(tds) > 1:
                shift['period'] = tds[1].get_text(strip=True)
            else:
                continue  # Skip if not enough cells

            if len(tds) > 2:
                shift['player_cnt'] = tds[2].get_text(strip=True)

            if len(tds) > 3 and tds[3].find('br'):
                # Split the content on the <br/> tag, preserving the formatting
                time_parts = tds[3].get_text(separator=' ').split()
                if len(time_parts) >= 2:
                    shift['elapsed_time'] = time_parts[0]
                    shift['game_time'] = time_parts[1]
                else:
                    shift['elapsed_time'] = ''
                    shift['game_time'] = ''
            else:
                shift['elapsed_time'] = ''
                shift['game_time'] = ''

            if len(tds) > 4:
                shift['event_type'] = tds[4].get_text(strip=True)
            else:
                shift['event_type'] = ''

            if len(tds) > 5:
                shift['desc'] = tds[5].get_text(strip=True)
            else:
                shift['desc'] = ''

            if len(tds) < 11 or (shift['event_type'] in ['PSTR', 'PGEND', 'GEND', 'ANTHEM']):
                shift['away_players'] = []
                shift['home_players'] = []
                shifts.append(shift)
                continue
            else:
                try:
                    num_away_players = len(tds[6].find_all('table')) - 1
                    away_players = []
                    away_player_list = [7 + (4 * x) for x in range(0, num_away_players)]

                    for away_player in away_player_list:
                        if away_player < len(tds):
                            sweater_number, position = split_data(tds[away_player].get_text(strip=True))
                            away_players.append((sweater_number, position))
                    shift['away_players'] = away_players

                    if not away_player_list:
                        home_tds_ind = 3
                    else:
                        home_tds_ind = max(away_player_list) + 3

                    if home_tds_ind < len(tds):
                        num_home_players = len(tds[home_tds_ind].find_all('table')) - 1
                        home_players = []
                        home_player_list = [home_tds_ind + 1 + (4 * x) for x in range(0, num_home_players)]

                        for home_player in home_player_list:
                            if home_player < len(tds):
                                sweater_number, position = split_data(tds[home_player].get_text(strip=True))
                                home_players.append((sweater_number, position))
                        shift['home_players'] = home_players
                    else:
                        shift['home_players'] = []
                except Exception as e:
                    print(f"Error processing player data in shift: {e}")
                    shift['away_players'] = []
                    shift['home_players'] = []

                shifts.append(shift)

    return shifts


def process_plays(data):
    """
    Process play data from the plays API endpoint.
    Returns a list of play events with detailed information.
    """
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

        # Initialize all potential fields to None
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

        # Process specific play types
        if play.get('typeCode') == 502:  # Faceoff
            shift['faceoff_winner'] = play['details'].get('winningPlayerId', None)
            shift['faceoff_loser'] = play['details'].get('losingPlayerId', None)
        elif play.get('typeCode') == 503:  # Hit
            shift['hitting_player'] = play['details'].get('hittingPlayerId', None)
            shift['hittee_player'] = play['details'].get('hitteePlayerId', None)
        elif play.get('typeCode') == 504:  # Giveaway
            shift['giveaway'] = play['details'].get('playerId', None)
        elif play.get('typeCode') == 505:  # Goal
            shift['goal'] = play['details'].get('scoringPlayerId', None)
            shift['shot_attempt'] = play['details'].get('scoringPlayerId', None)
            shift['shot_on_goal'] = play['details'].get('scoringPlayerId', None)
            shift['goal_assist1'] = play['details'].get('assist1PlayerId', None)
            shift['goal_assist2'] = play['details'].get('assist2PlayerId', None)
            shift['goal_against'] = play['details'].get('goalieInNetId', None)
        elif play.get('typeCode') == 506:  # Shot
            shift['shot_attempt'] = play['details'].get('shootingPlayerId', None)
            shift['shot_on_goal'] = play['details'].get('shootingPlayerId', None)
            shift['shot_saved'] = play['details'].get('goalieInNetId', None)
        elif play.get('typeCode') == 507:  # Missed shot
            shift['missed_shot_attempt'] = play['details'].get('shootingPlayerId', None)
        elif play.get('typeCode') == 508:  # Blocked shot
            shift['blocked_shot_attempt'] = play['details'].get('shootingPlayerId', None)
            shift['blocked_shot_saved'] = play['details'].get('blockingPlayerId', None)
        elif play.get('typeCode') == 509:  # Penalty
            if play['details'].get('duration', None) is not None:
                if play['details'].get('typeCode', None) == 'MAT':
                    shift['penalty_duration'] = play['details'].get('duration', None) - 10
                else:
                    shift['penalty_duration'] = play['details'].get('duration', None)
            if play['details'].get('committedByPlayerId', None) is not None:
                shift['penalty_committed'] = play['details'].get('committedByPlayerId', None)
                shift['penalty_drawn'] = play['details'].get('drawnByPlayerId', None)
                shift['penalty_served'] = play['details'].get('servedByPlayerId', None)
            elif play['details'].get('servedByPlayerId', None) is not None:
                shift['penalty_served'] = play['details'].get('servedByPlayerId', None)
        elif play.get('typeCode') == 516:  # Stoppage
            shift['stoppage'] = play['details'].get('reason', None)
            if shift['stoppage'] not in ['icing', 'goalie-stopped-after-sog', 'puck-in-crowd', 'puck-in-netting',
                                         'offside', 'puck-in-crowd', 'puck-in-benches', 'puck-frozen', 'tv-timeout',
                                         'high-stick', 'net-dislodged-defensive-skater', 'player-injury',
                                         'video-review',
                                         'referee-or-linesman', 'clock-problem',
                                         'hand-pass', 'objects-on-ice', 'goalie-puck-frozen-played-from-beyond-center',
                                         'visitor-timeout', 'net-dislodged-offensive-skater',
                                         'chlg-hm-goal-interference',
                                         'chlg-vis-goal-interference', 'chlg-hm-missed-stoppage',
                                         'skater-puck-frozen', 'ice-scrape', 'chlg-league-goal-interference',
                                         'player-equipment', 'chlg-hm-off-side', 'chlg-vis-off-side',
                                         'chlg-hm-missed-stoppage', 'home-timeout', 'chlg-vis-missed-stoppage',
                                         'puck-in-penalty-benches', 'ice-problem', 'net-dislodged-by-goaltender',
                                         'rink-repair', 'chlg-league-missed-stoppage', 'official-injury',
                                         'chlg-hm-puck-over-glass',
                                         'chlg-league-off-side', 'switch-sides']:
                print(f'\n')
                print(f'collect play stoppage reason: {play["details"]["reason"]}')
                print(f'\n')
        elif play.get('typeCode') == 521:  # Period end
            shift['period_end'] = True
        elif play.get('typeCode') == 523:  # Period end (shootout)
            shift['period_end'] = True
        elif play.get('typeCode') == 525:  # Takeaway
            shift['takeaway'] = play['details'].get('playerId', None)
        elif play.get('typeCode') == 535:  # Delayed penalty
            shift['delayed_penalty'] = 1
        elif play.get('typeCode') == 537:  # Penalty shot missed
            shift['penalty_shot'] = play['details'].get('shootingPlayerId', None)
            shift['penalty_shot_saved'] = play['details'].get('goalieInNetId', None)

        plays.append(shift)

    return plays


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