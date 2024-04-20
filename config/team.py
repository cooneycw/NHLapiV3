class Team:
    _teams = set()

    def __init__(self, team_id):
        self._teams.add(self)
        self.team_id = team_id

    @classmethod
    def get_teams(cls):
        teams = [(team.team_id, team) for team in list(cls._teams)]
        sorted_tuples = sorted(teams, key=lambda x: x[0])
        return sorted_tuples

