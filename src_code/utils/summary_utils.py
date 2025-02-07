import pandas as pd
import networkx as nx


def create_combined_summary_excel(graph, filename='combined_summary.xlsx'):
    """
    Create an Excel workbook with a single sheet that lists, for each team,
    a team total row followed by rows for each player on that team.
    """
    # 1. Extract Player Data from the graph.
    # Adjust the keys as needed if your node attributes differ.
    player_records = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") == "player":
            record = {
                "Player_ID": node,
                "Team": attrs.get("team", ""),
                "Games_Played": attrs.get("games_played", 0),
                "TOI": sum(attrs.get("toi", [0, 0, 0])),
                "Faceoff_Taken": sum(attrs.get("faceoff_taken", [0, 0, 0])),
                # Adjust the key below if necessary (e.g., "faceofff_wons" might be a typo)
                "Faceoff_Won": sum(attrs.get("faceofff_wons", [0, 0, 0])),
                "Shot_On_Goal": sum(attrs.get("shot_on_goal", [0, 0, 0])),
                "Shot_Saved": sum(attrs.get("shot_saved", [0, 0, 0])),
                "Goal": sum(attrs.get("goal", [0, 0, 0])),
                "Hit_Another": sum(attrs.get("hit_another_player", [0, 0, 0])),
                "Hit_By": sum(attrs.get("hit_by_player", [0, 0, 0])),
                "Penalties": sum(attrs.get("penalties_duration", [0, 0, 0]))
            }
            player_records.append(record)

    df_players = pd.DataFrame(player_records)

    # 2. Compute Team Totals by grouping player stats
    team_totals = df_players.groupby("Team", as_index=False).agg({
        "Games_Played": "sum",
        "TOI": "sum",
        "Faceoff_Taken": "sum",
        "Faceoff_Won": "sum",
        "Shot_On_Goal": "sum",
        "Shot_Saved": "sum",
        "Goal": "sum",
        "Hit_Another": "sum",
        "Hit_By": "sum",
        "Penalties": "sum"
    })
    # Mark these rows as team totals
    team_totals["Player_ID"] = "Team Total"

    # 3. Build a Combined DataFrame with Team Totals and then the player details.
    combined_rows = []
    # Get a sorted list of teams (or choose your desired order)
    teams = sorted(df_players["Team"].unique())
    for team in teams:
        # Add the team total row
        team_total_row = team_totals[team_totals["Team"] == team]
        combined_rows.append(team_total_row)
        # Add the player rows for this team
        team_players = df_players[df_players["Team"] == team]
        combined_rows.append(team_players)
        # Optionally, add a blank row as a separator (this row has empty strings or Nones)
        blank_row = pd.DataFrame([{
            "Team": "",
            "Player_ID": "",
            "Games_Played": None,
            "TOI": None,
            "Faceoff_Taken": None,
            "Faceoff_Won": None,
            "Shot_On_Goal": None,
            "Shot_Saved": None,
            "Goal": None,
            "Hit_Another": None,
            "Hit_By": None,
            "Penalties": None
        }])
        combined_rows.append(blank_row)

    # Concatenate all rows into one DataFrame
    df_combined = pd.concat(combined_rows, ignore_index=True)

    # 4. Write the combined DataFrame to a single Excel file.
    with pd.ExcelWriter(filename) as writer:
        df_combined.to_excel(writer, sheet_name="Team_Player_Summary", index=False)

    print(f"Combined summary Excel saved as {filename}")


# Example usage:
if __name__ == '__main__':
    # Assume that data_graph is the NetworkX graph you have built and updated.
    # For example, after calling your add_team_node, add_player_node, add_game, etc.
    data_graph = nx.Graph()

    # (Insert your graph-building code here to populate data_graph)

    # Once your graph is populated and stats are updated, create the Excel summary:
    create_combined_summary_excel(data_graph, filename='combined_summary.xlsx')
