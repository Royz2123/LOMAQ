import networkx as nx
import itertools
import numpy as np

import matplotlib.pyplot as plt


class DependencyGraph:
    def __init__(self, graph=None, num_agents=1, keep_cache=True):
        self.num_agents = num_agents

        # Save graph object
        self.graph = graph
        if graph is None:
            self.graph = DependencyGraph.create_default_graph(num_agents)

        # caching the neighborhoods. stored as double dict agent_index -> depth -> neighboring indicies
        self.cache = {}
        self.keep_cache = keep_cache
        if self.keep_cache:
            self.cache_nbrhds()

        # compute the max degree of the graph
        self.max_deg = self.compute_graph_deg()

    def get_adjacency_matrix(self):
        A = nx.to_numpy_matrix(self.graph)
        I = np.matrix(np.eye(A.shape[0]))
        return np.float32(A + I)

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

    # Default graph is a line graph between all the agents
    @staticmethod
    def create_default_graph(num_agents):
        graph = nx.Graph()
        graph.add_node(0)
        for i in range(num_agents - 1):
            graph.add_edge(i, i + 1)
        return graph


if __name__ == "__main__":
    graphobj = DependencyGraph(None, 9)
    print(graphobj.get_neighborhood(3))
    graphobj.display()
