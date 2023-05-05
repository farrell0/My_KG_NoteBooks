

#  This library file will give us our connection handle to Katana.
#  This file is not expected to be run stand-alone.
#
#  This file is referenced in;
#     .  L02_schema.py       where we run CRUD routines, aka run Python DAOs.
#     .  P00_web_server.py   where we maintain a session state.
#
#
#  A Jupyter NoteBook in this project is used to create and populate 
#  the graph before you run any of these files.


from katana import remote


####################################################


my_client = remote.Client()
   #
DB_NAME         = "my_db"
GRAPH_NAME      = "my_graph"

my_graph, *_ = my_client.get_database(name = DB_NAME).find_graphs_by_name(GRAPH_NAME)

print()
print(my_graph)
print()

