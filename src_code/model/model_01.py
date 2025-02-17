from src_code.utils.utils import load_game_data, create_player_dict
from src_code.utils.graph_utils import (
    create_graph, add_team_node, add_player_node, add_game, process_games_chronologically,
    add_player_game_performance, update_tgp_stats, update_pgp_stats, update_pgp_edge_stats,
    update_game_outcome)
from src_code.utils.summary_utils import update_game_nodes
from src_code.utils.display_graph_utils import visualize_game_graph
import copy


def model_data(config):
    dimension_names = "all_names"
    dimension_teams = "all_teams"
    dimension_shifts = "all_shifts"
    dimension_plays = "all_plays"
    dimension_game_rosters = "all_game_rosters"
    dimension_games = "all_boxscores"
    dimension_players = "all_players"

    data_names = config.load_data(dimension_names)
    data_games = config.load_data(dimension_games)
    data_teams = config.load_data(dimension_teams)
    data_players = config.load_data(dimension_players)
    data_shifts = config.load_data(dimension_shifts)
    data_plays = config.load_data(dimension_plays)
    data_game_roster = config.load_data(dimension_game_rosters)

    player_list, player_dict = create_player_dict(data_names)

    # input_graph = create_graph()
    data_graph = create_graph()

    for i, team in enumerate(data_teams):
        add_team_node(data_graph, team)

    for j, player in enumerate(player_list):
        add_player_node(data_graph, player, player_dict)

    process_games_chronologically(data_graph, data_games)

    team_game_maps = []
    for l, roster in enumerate(data_game_roster):
        team_game_map = add_player_game_performance(data_graph, roster)
        team_game_maps.append(team_game_map)

    print(data_graph)
    shifts = []

    for m, game in enumerate(data_games):
        verbose = False
        if (m % 10 == 0) and (m != 0):
            print(f'game {m} of {len(data_games)}')
            verbose = True
        # show_single_game_trimmed(data_graph, game['id'])
        shift_data = load_game_data(config.file_paths["game_output_pkl"] + f'{str(game["id"])}')

        # for q, player_dat in enumerate(shift_data['player_data']):
        #
        #     goal_instances = find_goals(player_dat)
        #     for goal in goal_instances:
        #         print(
        #             f"Goal found - shift index: {q} Player Index: {goal['player_index']}, Player ID: {goal['player_id']}, Team: {goal['player_team']}, Periods: {goal['goal_periods']}")

        process_shift_data(data_graph, verbose, team_game_maps[m], shift_data)
        shifts.append(shift_data)
        update_game_outcome(data_graph, game['id'], game)

        # for player in data_game_roster[m]:
        #     update_player_temporal_features(
        #         data_graph,
        #         player['player_id'],
        #         str(game['id']),
        #         game['game_date']
        #     )

        # Create visualization after all game data is processed and updated
        if m % 10 == 0:
            visualize_game_graph(data_graph, game['id'],
                                 output_path=f"{config.file_paths['game_output_jpg']}/game_{game['id']}_network_{game['game_date']}.jpg")

    cwc = 0


def process_shift_data(data_graph, verbose, team_game_map, shift_data):
    # called on a per-game basis
    game_id = shift_data["game_id"]
    game_date = shift_data["game_date"]
    away_team = shift_data["away_teams"]
    home_team = shift_data["home_teams"]
    period_id = shift_data["period_id"]
    period_code = shift_data["period_code"]
    time_index = shift_data["time_index"]
    time_on_ice = shift_data["time_on_ice"]
    event_id = shift_data["event_id"]
    shift_id = shift_data["shift_id"]
    away_empty_net = shift_data["away_empty_net"]
    home_empty_net = shift_data["home_empty_net"]
    away_skaters = shift_data["away_skaters"]
    home_skaters = shift_data["home_skaters"]
    player_data = shift_data["player_data"]


    for i, shift in enumerate(shift_id):
        if (i % 100 == 0) and (i != 0) and verbose == True:
            print(f'{i} of {len(shift_id)}')
        # one per shift
        team_map = {}
        line_player_team_map = {}
        for player_dat in player_data[i]:
            if player_dat['player_team'] not in team_map:
                team_map[player_dat['player_team']] = (game_id[i], player_dat['player_team'])
            game_team = (game_id[i], player_dat['player_team'])
            if game_team not in line_player_team_map:
                line_player_team_map[game_team] = []
            line_player_team_map[game_team].append(player_dat['player_id'])

        for team in line_player_team_map:
            other_players = copy.deepcopy(line_player_team_map[team])
            for j, player_dat in enumerate(player_data[i]):
                if player_dat['player_id'] not in line_player_team_map[team]:
                    continue
                other_players.remove(player_dat['player_id'])
                team_tgp = str(team[0]) + '_' + team[1]
                player_pgp = str(team[0]) + '_' + str(player_dat['player_id'])
                # if player_data[i][j]['goal'][period_code[i]] == 1:
                #     print(j, player_dat['player_id'])
                #     print(line_player_team_map[team])
                #     print(f"{i}:{j}:{player_data[i][j]['player_team']}:{player_data[i][j]['goal']}")
                #     print('\n')
                update_tgp_stats(data_graph, team_tgp, period_code[i], player_dat)
                update_pgp_stats(data_graph, player_pgp, period_code[i], player_dat)
                for k, other in  enumerate(other_players):
                    other_pgp = str(team[0]) + '_' + str(other)
                    update_pgp_edge_stats(data_graph, player_pgp, other_pgp, period_code[i], player_dat)


def find_goals(player_data):
    goals = []
    for i, shift_player in enumerate(player_data):
        # Check if any period has a goal
        if 1 in shift_player['goal']:
            goals.append({
                'player_index': i,
                'player_id': shift_player['player_id'],
                'player_team': shift_player['player_team'],
                'goal_periods': [p + 1 for p, goal in enumerate(shift_player['goal']) if goal == 1]
            })
    return goals

