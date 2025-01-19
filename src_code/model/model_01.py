from src_code.utils.utils import load_game_data, create_player_dict
from src_code.utils.graph_utils import create_graph, show_single_game_trimmed, add_team_node, add_player_node, add_game, add_player_game_performance
import networkx as nx


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

    for l, roster in enumerate(data_game_roster):
        add_player_game_performance(data_graph, roster)

    print(data_graph)
    shifts = []

    for m, game in enumerate(data_games):
        # show_single_game_trimmed(target_graph, game['id'])
        shift_data = load_game_data(config.file_paths["game_output_pkl"] + f'{str(game["id"])}')
        process_shift_data(data_graph, shift_data)
        shifts.append(shift_data)
    cwc = 0


def process_shift_data(data_graph, shift_data):
    game_id = shift_data["game_id"]
    game_date = shift_data["game_date"]
    away_team = shift_data["away_teams"]
    home_team = shift_data["home_teams"]
    period_id = shift_data["period_id"]
    time_index = shift_data["time_index"]
    time_on_ice = shift_data["time_on_ice"]
    event_id = shift_data["event_id"]
    shift_id = shift_data["shift_id"]
    away_empty_net = shift_data["away_empty_net"]
    home_empty_net = shift_data["home_empty_net"]
    away_skaters = shift_data["away_skaters"]
    home_skaters = shift_data["home_skaters"]
    player_data = shift_data["player_data"]

    for i, shift in shift_id:
        cwc = 0

