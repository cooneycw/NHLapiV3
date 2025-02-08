from bs4 import BeautifulSoup
from datetime import datetime
from src_code.utils.utils import create_dummy_player
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests


def get_season_data(config):
    print(f'Gathering season data...')
    if config.delete_files:
        config.del_data()
    dimension = "all_seasons"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_seasons is False:
        for season in prior_data:
            _ = config.Season(season)
    if (prior_data is None) or (config.reload_seasons is True):
        final_url = config.get_endpoint("seasons")
        response = requests.get(final_url)
        season_data = response.json()
        for season in season_data:
            _ = config.Season(season)
        config.save_data(dimension, season_data)


def get_team_list(config):
    print(f'Gathering team data...')
    dimension = "all_teams"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_teams is False:
        for team in prior_data:
            _ = config.Team(team)
    if (prior_data is None) or (config.reload_teams is True):
        final_url = config.get_endpoint("standings")
        response = requests.get(final_url)
        standings_data = response.json()
        team_list = []
        for item in standings_data['standings']:
            team_abbrev = item['teamAbbrev']['default']
            team_list.append(team_abbrev)
        team_list = sorted(team_list)
        for team in team_list:
            config.Team(team)
        config.save_data(dimension, team_list)


# def get_rosters(config):
#     tasks = []
#     async with aiohttp.ClientSession() as session:
#         for team in config.get_teams():
#             url = f"{config.endpoints['roster']}"
#             final_url = url.format(base_url=config.base_url, team=team)
#             task = asyncio.create_task(fetch_roster_data(session, final_url, team))
#             tasks.append(task)
#         roster_list = await asyncio.gather(*tasks)
#     config.set_rosters(roster_list)
#
#
# def get_player_list(config):
#     roster_data = config.get_rosters()
#     player_list = []
#     for team, roster_info in roster_data:
#         for player_type in roster_info.keys():
#             for player in roster_info[player_type]:
#                 identity = (team, player['id'], player_type, player['lastName'], player['firstName'])
#                 player_list.append(identity)
#     config.set_player_list(player_list)


def fetch_games_for_team_and_season(config, season, team):
    """
    Worker function to fetch games for a single (season, team) pair.
    Returns a list of tuples (game_id, game_date, home_team, away_team, season).
    """
    final_url = config.get_endpoint("schedule", team=team[0], season=season[0])
    response = requests.get(final_url)
    response.raise_for_status()  # Ensure we raise an exception if there's a bad response

    data = response.json()
    partial_data = []

    for game in data["games"]:
        # Filter out unwanted game types
        if game["gameType"] != 2:
            continue

        # Filter out postponed game
        if game['gameDate'] == '2025-01-08':  # postponed game due to wildfires
            if game['awayTeam']['abbrev'] == 'CGY':
                continue

        game_date = datetime.strptime(game["gameDate"], "%Y-%m-%d")
        home_team = game["homeTeam"]['abbrev']
        away_team = game["awayTeam"]['abbrev']

        # Create the game object (if your create_game modifies shared state,
        # ensure that is thread-safe or refactor to store data locally)
        game_obj = config.Game.create_game(game['id'], game_date, home_team, away_team, season[0])
        if game_obj is None:
            continue

        partial_data.append(game_obj.get_game_tuple())

    return partial_data


def get_game_list(config):
    print('Gathering game data by team...')
    dimension = "all_games"
    prior_data = config.load_data(dimension)

    # If prior_data exists and reload isn't required, just rebuild existing game objects
    if prior_data and config.reload_games is False:
        for game in prior_data:
            _ = config.Game.create_game(game[0], game[1], game[2], game[3], game[4])
        return  # No need to re-fetch data

    # Otherwise, fetch the schedule data in parallel
    save_data = []

    # Collect all (season, team) pairs we need to process
    season_team_pairs = []
    for season in config.Season.get_selected_seasons(config.season_count):
        for team in config.Team.get_teams():
            season_team_pairs.append((season, team))

    # Use ThreadPoolExecutor for parallel network requests
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit a job for each (season, team) pair
        futures = [
            executor.submit(fetch_games_for_team_and_season, config, season, team)
            for (season, team) in season_team_pairs
        ]

        # As each future completes, add its returned data to save_data
        for future in as_completed(futures):
            result = future.result()  # May raise if any request failed
            save_data.extend(result)

    # Finally, persist the combined data (e.g., writing to a DB, file, etc.)
    config.save_data(dimension, save_data)


# def get_game_list(config):
#     print(f'Gathering game data by team...')
#     dimension = "all_games"
#     prior_data = config.load_data(dimension)
#     if prior_data and config.reload_games is False:
#         for game in prior_data:
#             _ = config.Game.create_game(game[0], game[1], game[2], game[3], game[4])
#     if (prior_data is None) or (config.reload_games is True):
#         save_data = []
#         for season in config.Season.get_selected_seasons(config.season_count):
#             for team in config.Team.get_teams():
#                 print(f'Getting games for {season[0]}:{team[0]}')
#                 final_url = config.get_endpoint("schedule", team=team[0], season=season[0])
#                 response = requests.get(final_url)
#                 response.raise_for_status()  # Ensure the response is OK
#                 data = response.json()
#                 for game in data["games"]:
#                     if game["gameType"] != 2:
#                         continue
#                     if game['gameDate'] == '2025-01-08':  # postponed game due to wildfires
#                         if game['awayTeam']['abbrev'] == 'CGY':
#                             continue
#                     game_date = datetime.strptime(game["gameDate"], "%Y-%m-%d")
#                     home_team = game["homeTeam"]['abbrev']
#                     away_team = game["awayTeam"]['abbrev']
#                     game_obj = config.Game.create_game(game['id'], game_date, home_team, away_team, season[0])
#                     if game_obj is None:
#                         continue
#                     save_data.append(game_obj.get_game_tuple())
#
#         # Set the games in the config, assuming this modifies some shared state or configuration
#         config.save_data(dimension, save_data)


import concurrent.futures
import requests


def process_game_boxscore(config, game):
    """
    Function to fetch and process a single game's boxscore data.
    Returns (game_results, player_results) tuples for each game.
    """
    game_id = game[0]
    game_date = game[1]

    # Build and fetch Boxscore V1
    final_url_v1 = config.get_endpoint("boxscore_v1", game_id=game_id)
    response_v1 = requests.get(final_url_v1)
    response_v1.raise_for_status()
    data_v1 = response_v1.json()

    # Build and fetch Boxscore V2
    final_url_v2 = config.get_endpoint("boxscore_v2", game_id=game_id)
    response_v2 = requests.get(final_url_v2)
    response_v2.raise_for_status()
    data_v2 = response_v2.json()

    # Process results
    game_obj = config.Game.get_game(game_id)
    success, game_results = process_boxscore(game_obj.game_id, data_v1, data_v2)
    player_results_list = []

    if success:
        # We do NOT update game_obj or Player objects here if we want to avoid concurrency issues
        # We'll just return the data, and update them later in the main thread
        for i, team in enumerate(['awayTeam', 'homeTeam']):
            position_data = data_v1['playerByGameStats'][team]
            for position in ['forwards', 'defense', 'goalies']:
                players_data = position_data[position]
                for player in players_data:
                    if position != 'goalies':
                        player_results = process_forward_defense_data(player)
                    else:
                        player_results = process_goalie_data(player)
                    player_results['i'] = i
                    player_results['game'] = game
                    player_results_list.append(player_results)

        return (game_results, player_results_list)
    else:
        # You can decide what to return in case of failure
        return (None, [])


def get_boxscore_list(config):
    print('Gathering boxscore data...')
    dimension_games = "all_boxscores"
    dimension_players = "all_players"
    prior_data_games = config.load_data(dimension_games)
    prior_data_players = config.load_data(dimension_players)

    # 1. Handle existing data if reload is not needed
    if prior_data_games and prior_data_players and config.reload_boxscores is False:
        for game_results in prior_data_games:
            game_obj = config.Game.get_game(game_results['id'])
            game_obj.update_game(game_results)
        for player_results in prior_data_players:
            player_obj = config.Player.create_player(player_results['player_id'])
            i = player_results['i']
            game = player_results['game']
            player_obj.update_player(game[0], game[1], game[3 - i], game[4], player_results)

    # 2. Otherwise, fetch and process new data
    if (prior_data_games is None) or (config.reload_boxscores is True):
        save_game_results = []
        save_player_results = []

        # Collect only the games we actually need to process
        games_to_process = [
            g for g in config.Game.get_games()
            if g[1].date() < config.curr_date
        ]

        # --- PARALLEL EXECUTION BLOCK ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit a job for each game
            future_to_game = {
                executor.submit(process_game_boxscore, config, game): game
                for game in games_to_process
            }

            # As futures complete, gather results
            for future in concurrent.futures.as_completed(future_to_game):
                game = future_to_game[future]
                try:
                    game_results, player_results_list = future.result()
                    if game_results:
                        save_game_results.append(game_results)
                        # We'll update the config objects here in the main thread
                        game_obj = config.Game.get_game(game_results['id'])
                        game_obj.update_game(game_results)

                        # Now process each player's results
                        for player_results in player_results_list:
                            player_obj = config.Player.create_player(player_results['player_id'])
                            i = player_results['i']
                            g = player_results['game']
                            player_obj.update_player(g[0], g[1], g[3 - i], g[4], player_results)
                            save_player_results.append(player_results)

                except Exception as exc:
                    # Handle exceptions as needed
                    print(f"Exception while processing game {game[0]}: {exc}")

        # Finally, save all data
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


def get_playbyplay_data(config):
    print(f'Gathering play by play data...')

    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"

    # Load prior data
    prior_data_shifts = config.load_data(dimension_shifts)
    prior_data_plays = config.load_data(dimension_plays)
    prior_data_game_rosters = config.load_data(dimension_game_rosters)

    # If we already have data and reload is off, do nothing
    if prior_data_shifts and config.reload_playbyplay is False:
        return  # Or do any logic you want here

    # Otherwise, fetch new data in parallel
    save_results_shifts = []
    save_results_plays = []
    save_results_game_rosters = []

    games = config.Game.get_games()

    # Filter out future games or any that you want to skip
    games_to_process = [
        g for g in games
        if g[1].date() < config.curr_date  # e.g. skip if game_date >= curr_date
    ]

    # Use a context manager for the executor, setting max_workers as desired
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Create a dict of future -> game so we know which game each future corresponds to
        future_to_game = {
            executor.submit(process_single_game, config, game): game
            for game in games_to_process
        }

        # As each thread completes, gather its data
        for future in as_completed(future_to_game):
            game = future_to_game[future]
            try:
                results_shifts, results_plays, results_game_rosters = future.result()
                save_results_shifts.append(results_shifts)
                save_results_plays.append(results_plays)
                save_results_game_rosters.append(results_game_rosters)
            except Exception as e:
                # Handle exceptions (e.g., log or re-raise)
                print(f"Error processing game {game}: {e}")

    # Once all tasks are done, save results
    config.save_data(dimension_shifts, save_results_shifts)
    config.save_data(dimension_plays, save_results_plays)
    config.save_data(dimension_game_rosters, save_results_game_rosters)


def process_single_game(config, game):
    """
    Fetch shifts, plays, and roster data for a single game and return the results.
    """
    game_id, game_date, _, _ = game

    print(f'Getting shift data for {game_id}:{game_date}...')

    # 1. SHIFT DATA
    game_obj = config.Game.get_game(game_id)
    final_url = game_obj.playbyplay
    response = requests.get(final_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    results_shifts = process_shifts(config, soup)

    # 2. PLAYS + GAME ROSTERS
    final_url_plays = config.get_endpoint("plays", game_id=game_id)
    response_plays = requests.get(final_url_plays)
    response_plays.raise_for_status()
    data_plays = response_plays.json()

    results_plays = process_plays(data_plays)
    results_game_rosters = process_game_rosters(data_plays)

    return results_shifts, results_plays, results_game_rosters


# def get_playbyplay_data(config):
#     print(f'Gathering play by play data...')
#     dimension_shifts = "all_shifts"
#     dimension_plays = "all_plays"
#     dimension_game_rosters = "all_game_rosters"
#     prior_data_shifts = config.load_data(dimension_shifts)
#     prior_data_plays = config.load_data(dimension_plays)
#     prior_data_game_rosters = config.load_data(dimension_game_rosters)
#     if prior_data_shifts and config.reload_playbyplay is False:
#         for results in prior_data_plays:
#             cwc = 0
#     if (prior_data_shifts is None) or (config.reload_playbyplay is True):
#         save_results_shifts = []
#         save_results_plays = []
#         save_results_game_rosters = []
#         for game in config.Game.get_games():
#             if game[1].date() >= config.curr_date:
#                 continue
#             print(f'Getting shift data for {game[0]}:{game[1]}:{game[2]}:{game[3]}')
#             game_obj = config.Game.get_game(game[0])
#             final_url = game_obj.playbyplay
#             response = requests.get(final_url)
#             response.raise_for_status()  # Ensure the response is OK
#             soup = BeautifulSoup(response.text, 'html.parser')
#             results_shifts = process_shifts(config, soup)
#
#             final_url_plays = config.get_endpoint("plays", game_id=game[0])
#             response_plays = requests.get(final_url_plays)
#             response_plays.raise_for_status()  # Ensure the response is OK
#             data_plays = response_plays.json()
#             results_plays = process_plays(data_plays)
#             results_game_rosters = process_game_rosters(data_plays)
#
#             # game_obj = config.Game.get_game(game[0])
#
#             save_results_shifts.append(results_shifts)
#             save_results_plays.append(results_plays)
#             save_results_game_rosters.append(results_game_rosters)
#             # game_obj.update_game(results)
#         config.save_data(dimension_shifts, save_results_shifts)
#         config.save_data(dimension_plays, save_results_plays)
#         config.save_data(dimension_game_rosters, save_results_game_rosters)


def process_shifts(config, soup):
    print(f'Gathering player names...')
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


def process_game_rosters(data):
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
                                 'rink-repair', 'chlg-league-missed-stoppage', 'official-injury', 'chlg-hm-puck-over-glass']:
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


def get_player_names(config):
    print(f'Gathering player names...')
    dimension = "all_names"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_playernames is False:
        for player_data in prior_data:
            player_id = player_data['playerId']
            last_name = player_data['lastName']['default']
            first_name = player_data['firstName']['default']
            player_obj = config.Player.get_player(player_id)
            player_obj.update_name(last_name, first_name)

            # update player names
    if (prior_data is None) or (config.reload_playernames is True):
        save_data = []
        for i, player_id in enumerate(config.Player.get_players()):
            if i % 100 == 0:
                print(f'{i} players gathered..')
            final_url = config.get_endpoint("player", player_id=player_id)
            response = requests.get(final_url)
            response.raise_for_status()  # Ensure the response is OK
            player_data = response.json()

            last_name = player_data['lastName']['default']
            first_name = player_data['firstName']['default']
            player_obj = config.Player.get_player(player_id)
            player_obj.update_name(last_name, first_name)
            save_data.append(player_data)

        # Set the games in the config, assuming this modifies some shared state or configuration
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