import pandas as pd
import networkx as nx


def update_game_nodes(graph):
    for node, data in graph.nodes(data=True):
        if data.get("type") == "game":
            cwc = 0

    return graph
