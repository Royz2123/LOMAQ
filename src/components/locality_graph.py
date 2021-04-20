import networkx as nx
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