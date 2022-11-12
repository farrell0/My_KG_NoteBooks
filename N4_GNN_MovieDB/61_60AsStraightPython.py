
from katana import remote
from katana.remote import import_data

my_client = remote.Client()

print(my_client)


DB_NAME         = "my_db"
GRAPH_NAME      = "my_graph"
   #
my_graph, *_ = my_client.get_database(name=DB_NAME).find_graphs_by_name(GRAPH_NAME)
   #
print(my_graph)


print(my_graph.num_nodes())
print(my_graph.num_edges())


############################################################################