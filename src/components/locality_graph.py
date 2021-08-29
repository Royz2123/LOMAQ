import networkx as nx
import itertools
import numpy as np
from scipy.linalg import fractional_matrix_power

import matplotlib.pyplot as plt


class DependencyGraph:
    def __init__(self, graph=None, num_agents=1, keep_cache=True):
        self.num_agents = num_agents

        # Save graph object, by default let it be a full graph
        self.graph = graph
        if graph is None:
            self.graph = DependencyGraph.build_simple_graph(num_agents, graph_type="full")

        # Create also a corresponding graph with self loops
        self.graph_self_loops = self.create_self_loop_graph()

        # caching the neighborhoods. stored as double dict agent_index -> depth -> neighboring indicies
        self.cache = {}
        self.keep_cache = keep_cache
        if self.keep_cache:
            self.cache_nbrhds()

        # compute the max degree of the graph
        self.max_deg = self.compute_graph_deg()

    def create_self_loop_graph(self):
        graph_self_loops = self.graph.copy()

        self_loops = []
        for i in range(self.graph.number_of_nodes()):
            self_loops.append((i, i))

        graph_self_loops.add_edges_from(self_loops)
        return graph_self_loops

    def get_adjacency_matrix(self, self_loops=True, normalize=True):
        if self_loops:
            adj_matrix = nx.to_numpy_matrix(self.graph_self_loops)
        else:
            adj_matrix = nx.to_numpy_matrix(self.graph)

        if normalize:
            d_half_norm = self.get_normalizing_diag()
            adj_matrix = d_half_norm.dot(adj_matrix).dot(d_half_norm)

        return np.float32(adj_matrix)

    def get_normalizing_diag(self, power=-0.5):
        deg_matrix = self.graph_self_loops.degree()
        diag_deg_matrix = np.diag([deg for (n, deg) in list(deg_matrix)])
        diag_deg_matrix_inv = fractional_matrix_power(diag_deg_matrix, power)
        return diag_deg_matrix_inv

    # Function that basically finds n_j
    def find_reward_groups(self, l=1, beta2=1):
        all_connected_subgraphs = []

        for nb_nodes in range(1, l + 1):
            all_connected_subgraphs.append([
                list(selected_nodes)
                for selected_nodes in itertools.combinations(self.graph, nb_nodes)
                if (
                        nx.is_connected(self.graph.subgraph(selected_nodes))
                        and nx.diameter(self.graph.subgraph(selected_nodes)) < (2 * beta2 + 1)
                )
            ])

        return all_connected_subgraphs

    def compute_graph_deg(self):
        return max(self.graph.degree, key=lambda x: x[1])[1]

    def cache_nbrhds(self):
        for agent_index in range(self.num_agents):
            self.cache[agent_index] = {}

            # max depth is num_agents, can sometimes be smaller
            for depth in range(self.num_agents + 1):
                self.cache[agent_index][depth] = self.compute_nbrhood(agent_index, depth)

    def get_nbrhoods(self, depth=1):
        return [self.get_nbrhood(agent_index, depth) for agent_index in range(self.num_agents)]

    def get_nbrhood(self, agent_index, depth=1):
        if self.keep_cache:
            return self.cache[agent_index][depth]
        else:
            return self.compute_nbrhood(agent_index, depth)

    def compute_nbrhood(self, agent_index, depth=1):
        return list(nx.single_source_shortest_path_length(self.graph, agent_index, cutoff=depth).keys())

    def display(self):
        nx.draw_networkx(self.graph, arrows=True, with_labels=True)
        plt.show()

    @staticmethod
    def build_simple_graph(num_agents, graph_type="empty"):
        graph = nx.Graph()

        # add all the agents (necessary for the empty case)
        for i in range(num_agents):
            graph.add_node(i)

        # if not empty then full and add all the edges
        if graph_type == "full":
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    graph.add_edge(i, j)

        # If line graph
        if graph_type == "line":
            for i in range(num_agents - 1):
                graph.add_edge(i, i + 1)

        return graph


if __name__ == "__main__":
    graphobj = DependencyGraph(None, 9)
    print(graphobj.get_neighborhood(3))
    graphobj.display()
