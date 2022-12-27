

import pandas as pd
from tabulate import tabulate
   #
from katana import remote


# #############################################################


NUM_PARTITIONS  = 3
   #
DB_NAME         = "my_db"
GRAPH_NAME      = "my_graph"


# #############################################################


def f_init1():


   pd.set_option("display.width", 480)
   
   #  Sets horizontal scroll for wide outputs
   #
   from IPython.display import display, HTML
   display(HTML(""))

   return 


# #############################################################


def f_init2():
    
   #  Get a client handle
   #
   my_client = remote.Client()
   
   #  Connect to graph
   #
   my_graph, *_ = my_client.get_database(name=DB_NAME).find_graphs_by_name(GRAPH_NAME)
       
    
   return my_client, my_graph


# #############################################################


def f_init():
   f_init1()
   my_client, my_graph = f_init2()
    
   return my_client, my_graph





f_init2
