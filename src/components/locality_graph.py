import networkx as nx
import matplotlib.pyplot as plt


class DependencyGraph:
    def __init__(self, graph=None, num_agents=1):
        self.num_agents = num_agents

        # Save graph object
        self.graph = graph
        if graph is None:
            self.graph = DependencyGraph.create_default_graph(num_agents)

    def get_nbrhood(self, agent_index, depth=1):
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