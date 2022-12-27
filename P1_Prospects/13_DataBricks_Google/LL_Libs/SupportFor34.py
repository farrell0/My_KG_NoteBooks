

import pandas as pd
from tabulate import tabulate


def f_init():


   pd.set_option("display.width", 480)
   
   #  Sets horizontal scroll for wide outputs
   #
   from IPython.display import display, HTML
   display(HTML(""))


   NUM_PARTITIONS  = 3
      #
   DB_NAME         = "my_db"
   GRAPH_NAME      = "my_graph"
