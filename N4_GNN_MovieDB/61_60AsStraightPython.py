
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

l_rows = 40000

l_query  = """
   MATCH (n) 
   RETURN n.ol1 AS col1, n.col2 AS col2
   LIMIT {0}
   """.format(l_rows)

dd_result2 = my_graph.query(l_query)

print("")
l_cntr = 0
   #
for l_each in dd_result2.itertuples():
   l_cntr += 1  
      #
   if (l_cntr < 10):
      print(l_each)




