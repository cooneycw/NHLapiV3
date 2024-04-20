class PlayByPlay:
    _playbyplay = dict()

    def __init__(self, shift_id, input_dict):
        self.shift_id = shift_id
        self.input_dict = input_dict

    @classmethod
    def create_playbyplay(cls, shift_id, input_dict):
        if shift_id in cls._playbyplay:
            print(f"Shift with ID {shift_id}:{input_dict} already exists.")
            # tested - functions correctly.
            return None
        new_shift = cls(shift_id, input_dict)
        cls._playbyplay[shift_id] = new_shift
        return new_shift
