import pickle


def save_data(obj):
    with open('storage/config.pkl', 'wb') as outp:  # Use 'wb' to write in binary mode
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_data():
    # To load the object back
    with open('storage/config.pkl', 'rb') as inp:  # Use 'rb' to read in binary mode
        obj = pickle.load(inp)
    return obj

def period_time_to_game_time(period, time_str):
    """
    Converts a hockey period time (e.g., "2:05:30") to a game time index in minutes.

    Parameters:
    - period_time_str (str): Time in "period:mm:ss" format.

    Returns:
    - float: Game time index in minutes.

    Raises:
    - ValueError: If the input format is incorrect or values are out of expected ranges.
    """
    try:
        parts = time_str.strip().split(':')
        if len(parts) != 2:
            raise ValueError("Input must be in 'period:mm:ss' format.")

        minutes = int(parts[0])
        seconds = int(parts[1])

        if period < 1:
            raise ValueError("Period must be at least 1.")
        if not (0 <= minutes <= 20):
            raise ValueError("Minutes must be between 0 and 19.")
        if not (0 <= seconds < 60):
            raise ValueError("Seconds must be between 0 and 59.")

        # Calculate elapsed time
        elapsed_time = (period - 1) * 20 * 60
        elapsed_time += (20 - minutes * 60 - seconds)

        return elapsed_time
    except Exception as e:
        raise ValueError(f"Invalid input '{period}: {time_str}': {e}")


def game_time_to_period_time(game_time_seconds):
    """
    Converts a game time index in seconds to a hockey period time string "period:mm:ss".

    Parameters:
    - game_time_seconds (int): Game time index in seconds.

    Returns:
    - tuple: (period (int), time_str (str) in "mm:ss" format).

    Raises:
    - ValueError: If the game_time_seconds is out of expected range.
    """
    total_game_seconds = 60 * 60  # 60 minutes * 60 seconds
    if not (0 <= game_time_seconds <= total_game_seconds):
        raise ValueError("Game time must be between 0 and 3600 seconds (60 minutes).")

    period_length_seconds = 20 * 60  # Each period is 20 minutes

    period = (game_time_seconds // period_length_seconds) + 1
    if period > 3:
        period = 3  # Assuming standard 3 periods; adjust if overtime is considered

    time_into_period = game_time_seconds % period_length_seconds
    time_remaining_in_period = period_length_seconds - time_into_period

    minutes = time_remaining_in_period // 60
    seconds = time_remaining_in_period % 60

    return period, f"{int(minutes):02d}:{int(seconds):02d}"


def create_roster_dicts(data_game_roster, away_team, home_team):
    away_players = {}
    home_players = {}

    for player in data_game_roster:
        player_id = player['player_id']
        player_team = player['player_team']
        player_lname = player['player_last_name']['default']
        player_fname = player['player_first_name']['default']
        player_sweater = player['player_sweater']
        player_position = player['player_position']

        player_dat = {
            'player_id': player_id,
            'player_team': player_team,
            'player_lname': player_lname,
            'player_fname': player_fname,
            'player_sweater': player_sweater,
            'player_position': player_position,
        }
        if player['player_team'] == away_team:
            away_players[player_sweater] = player_dat
        else:
            home_players[player_sweater] = player_dat

    return away_players, home_players


def create_ordered_roster(data_game_roster, away_team, home_team):
    away_players = {}
    home_players = {}

    for player in data_game_roster:
        player_id = player['player_id']
        player_team = player['player_team']
        player_lname = player['player_last_name']['default']
        player_fname = player['player_first_name']['default']
        player_sweater = player['player_sweater']
        player_position = player['player_position']

        player_dat = {
            'player_id': player_id,
            'player_team': player_team,
            'player_lname': player_lname,
            'player_fname': player_fname,
            'player_sweater': player_sweater,
            'player_position': player_position,
        }
        if player['player_team'] == away_team:
            away_players[player_id] = player_dat
        else:
            home_players[player_id] = player_dat

    return away_players, home_players


def create_player_dict(data_names):
    player_list = []
    dict_of_players = {}
    for i, player in enumerate(data_names):
        player_id = player['playerId']
        retrieve_test = dict_of_players.get(player_id, False)
        if not retrieve_test:
            player_list.append(player_id)
            dict_of_players[player_id] = {
                'lastName': player['lastName']['default'],
                'firstName': player['firstName']['default'],
                'sweaterNumber': player.get('sweaterNumber', -1),
            }
        else:
            cwc = 0

    return player_list, dict_of_players


def create_player_stats(player_id):
    player_stats = {
        'player_id': player_id['player_id'],
        'player_team': player_id['player_team'],
        'player_position': player_id['player_position'],
        'toi': 0,
        'power_play': 0,
        'hit_another_player': 0,
        'hit_by_player': 0,
        'giveaways': 0,
        'takeaways': 0,
        'penalties': 0,
        'penalties_overtime': 0,
        'penalties_drawn': 0,
        'penalties_drawn_overtime': 0,
        'penalties_duration': 0,
        'penalties_duration_overtime': 0,
        'shot_attempt': 0,
        'shot_attempt_overtime': 0,
        'shot_attempt_shootout': 0,
        'shot_missed': 0,
        'shot_missed_overtime': 0,
        'shot_missed_shootout': 0,
        'shot_on_goal': 0,
        'shot_on_goal_overtime': 0,
        'shot_blocked': 0,
        'shot_blocked_overtime': 0,
        'shot_saved': 0,
        'shot_saved_overtime': 0,
        'shot_saved_shootout': 0,
        'faceoff_taken': 0,
        'faceoff_won': 0,
        'goal': 0,
        'goal_overtime': 0,
        'goal_shootout': 0,
        'assist': 0,
        'assist_overtime': 0,
        'goal_against': 0,
        'goal_against_overtime': 0,
        'goal_against_shootout': 0,
    }
    return player_stats
