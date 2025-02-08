import pandas as pd
import networkx as nx


def update_game_nodes(graph):
    for node, data in graph.nodes(data=True):
        if data.get("type") == "game":
            for u, v, data in graph.edges(data=True):
                if (graph.nodes[u]["type"] != "team_game_performance") and graph.nodes[v]["type"] != "team_game_performance":
                    continue
                cwc = 0
    return graph
