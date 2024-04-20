from bs4 import BeautifulSoup
from datetime import datetime
import requests


def get_season_data(config):
    if config.delete_files:
        config.del_data()
    dimension = "all_seasons"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_seasons is False:
        for season in prior_data:
            config.seasons.add(config.Season(season))
    if (prior_data is None) or (config.reload_seasons is True):
        final_url = config.get_endpoint("seasons")
        response = requests.get(final_url)
        season_data = response.json()
        for season in season_data:
            config.seasons.add(config.Season(season))
        config.save_data(dimension, season_data)


def get_team_list(config):
    dimension = "all_teams"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_teams is False:
        for team in prior_data:
            config.teams.add(config.Team(team))
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
            config.teams.add(config.Team(team))
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
#
#
def get_game_list(config):
    dimension = "all_games"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_games is False:
        for game in prior_data:
            game_obj = config.Game.create_game(game[0], game[1], game[2], game[3], game[4])
            config.games.add(game_obj)
    if (prior_data is None) or (config.reload_games is True):
        save_data = []
        for season in config.Season.get_selected_seasons(config.season_count):
            for team in config.Team.get_teams():
                print(f'Getting games for {season[0]}:{team[0]}')
                final_url = config.get_endpoint("schedule", team=team[0], season=season[0])
                response = requests.get(final_url)
                response.raise_for_status()  # Ensure the response is OK
                data = response.json()
                for game in data["games"]:
                    if game["gameType"] != 2:
                        continue
                    game_date = datetime.strptime(game["gameDate"], "%Y-%m-%d")
                    home_team = game["homeTeam"]['abbrev']
                    away_team = game["awayTeam"]['abbrev']
                    game_obj = config.Game.create_game(game['id'], game_date, home_team, away_team, season[0])
                    if game_obj is None:
                        continue
                    config.games.add(game_obj)
                    save_data.append(game_obj.get_game_tuple())

        # Set the games in the config, assuming this modifies some shared state or configuration
        config.save_data(dimension, save_data)


def get_boxscore_list(config):
    dimension = "all_boxscores"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_boxscores is False:
        for results in prior_data:
            game_obj = config.Game.get_game(results['id'])
            process_boxscore(game_obj, results)
    if (prior_data is None) or (config.reload_boxscores is True):
        save_results = []
        for game in config.Game.get_games():
            print(f'Getting boxscore for {game[0]}:{game[2]}:{game[3]}')
            final_url = config.get_endpoint("boxscore", game_id=game[0])
            response = requests.get(final_url)
            response.raise_for_status()  # Ensure the response is OK
            data = response.json()
            game_obj = config.Game.get_game(game[0])
            results = process_boxscore(game_obj, data)
            save_results.append(results)
            game_obj.update_game(results)
        # Set the games in the config, assuming this modifies some shared state or configuration
        config.save_data(dimension, save_results)


def process_boxscore(game_obj, data):
    results = dict()
    results['id'] = data['id']
    try:
        results['home_goals'] = data['homeTeam']['score']
    except Exception as e:
        print(f'Exception: {e}')
        cwc = 0
    results['away_goals'] = data['awayTeam']['score']
    results['play_by_play'] = data['summary']['gameReports']['playByPlay']
    for category in data['summary']['teamGameStats']:
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
    return results


def get_playbyplay_data(config):
    dimension = "all_shifts"
    prior_data = config.load_data(dimension)
    if prior_data and config.reload_playbyplay is False:
        for results in prior_data:
            game_obj = config.PlayByPlay.get_shift(results['shift_id'])
            process_boxscore(game_obj, results)
    if (prior_data is None) or (config.reload_playbyplay is True):
        save_results = []
        for game in config.Game.get_games():
            print(f'Getting shift data for {game[0]}:{game[2]}:{game[3]}')
            final_url = config.get_endpoint("boxscore", game_id=game[0])
            response = requests.get(final_url)
            response.raise_for_status()  # Ensure the response is OK
            data = response.json()
            game_obj = config.Game.get_game(game[0])
            results = process_boxscore(game_obj, data)
            save_results.append(results)
            game_obj.update_game(results)
        # Set the games in the config, assuming this modifies some shared state or configuration
        config.save_data(dimension, save_results)

# async def get_boxscore_list(config):
#     boxscore_list = []
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for game_id in config.get_games():
#             final_url = config.endpoints['boxscore'].format(base_url=config.base_url, game_id=game_id[2])
#             task = asyncio.create_task(fetch_boxscore_data(session, final_url, game_id))
#             tasks.append(task)
#         results = await asyncio.gather(*tasks)
#         boxscore_list.extend(results)
#     config.set_boxscores(boxscore_list)
#
#
# async def fetch_roster_data(session, url, team):
#     async with session.get(url) as response:
#         roster_data = await response.json()
#         return (team, roster_data)
#
#
# async def fetch_game_data(session, url, season, team):
#     async with session.get(url) as response:
#         schedule_data = await response.json()
#         return [(season, team, game['id'], game['gameDate']) for game in schedule_data['games'] if game['gameType'] == 2]
#
#
# async def fetch_boxscore_data(session, url, game_id):
#     async with session.get(url) as response:
#         boxscore_data = await response.json()
#         return (game_id[0], game_id[1], game_id[2], game_id[3], boxscore_data)
