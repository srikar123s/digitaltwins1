import networkx as nx

def build_downhill_graph(grid, mean_elevation):

    G = nx.DiGraph()
    G.add_nodes_from(range(len(grid)))

    spatial_index = grid.sindex

    for i, geom in enumerate(grid.geometry):

        neighbors = list(spatial_index.intersection(geom.bounds))

        for j in neighbors:
            if i == j:
                continue

            # 🚀 FAST: removed expensive geometry check
            if mean_elevation[i] > mean_elevation[j]:
                G.add_edge(i, j)

    return G