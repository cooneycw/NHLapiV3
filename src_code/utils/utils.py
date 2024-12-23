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

        minutes = int(parts[1])
        seconds = int(parts[2])

        if period < 1:
            raise ValueError("Period must be at least 1.")
        if not (0 <= minutes <= 20):
            raise ValueError("Minutes must be between 0 and 19.")
        if not (0 <= seconds < 60):
            raise ValueError("Seconds must be between 0 and 59.")

        # Calculate elapsed time
        elapsed_time = (period - 1) * 20
        elapsed_time += (20 - minutes - seconds / 60)

        return elapsed_time
    except Exception as e:
        raise ValueError(f"Invalid input '{period}: {time_str}': {e}")


def game_time_to_period_time(game_time):
    """
    Converts a game time index in minutes to a hockey period time string "period:mm:ss".

    Parameters:
    - game_time (float or int): Game time index in minutes.

    Returns:
    - str: Time in "period:mm:ss" format.

    Raises:
    - ValueError: If the game_time is out of expected range.
    """
    if not (0 <= game_time <= 60):
        raise ValueError("Game time must be between 0 and 60 minutes.")

    period_length = 20  # Each period is 20 minutes
    period = int(game_time // period_length) + 1

    time_in_period = period_length - (game_time % period_length)

    minutes = int(time_in_period)
    seconds = int(round((time_in_period - minutes) * 60))

    # Handle cases where rounding seconds might make it 60
    if seconds == 60:
        minutes += 1
        seconds = 0
        if minutes > 20:
            minutes = 20
            seconds = 0

    return period, f"{minutes:02d}:{seconds:02d}"

def create_player_dict(data_players):
    dict_of_players = {}
    cwc =0
    return dict_of_players