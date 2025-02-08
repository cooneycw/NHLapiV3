from src_code.utils.utils import load_game_data, create_player_dict
from src_code.utils.graph_utils import create_graph, show_single_game_trimmed, add_team_node, add_player_node, add_game, \
    add_player_game_performance, update_tgp_stats, update_pgp_stats, update_pgp_edge_stats
from src_code.utils.summary_utils import update_game_nodes
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

    for k, game in enumerate(data_games):
        add_game(data_graph, game)

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
        process_shift_data(data_graph, verbose, team_game_maps[m], shift_data)
        shifts.append(shift_data)

    data_graph = update_game_nodes(data_graph)


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

        for j, team in enumerate(line_player_team_map):
            other_players = copy.deepcopy(line_player_team_map[team])
            for player in line_player_team_map[team]:
                other_players.remove(player)
                team_tgp = str(team[0]) + '_' + team[1]
                player_pgp = str(team[0]) + '_' + str(player)
                update_tgp_stats(data_graph, team_tgp, period_code[i], player_data[i][j])
                update_pgp_stats(data_graph, player_pgp, period_code[i], player_data[i][j])
                for k, other in  enumerate(other_players):
                    other_pgp = str(team[0]) + '_' + str(other)
                    update_pgp_edge_stats(data_graph, player_pgp, other_pgp, period_code[i], player_data[i][j])

