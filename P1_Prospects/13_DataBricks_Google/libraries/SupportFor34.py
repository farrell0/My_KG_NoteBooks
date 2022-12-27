
#  Push the boilerplate code to a supporting file
#


#  # #############################################################
#  # #############################################################


def f_init1():
    
   import pandas as pd
   from tabulate import tabulate


   pd.set_option("display.width", 480)
   
   #  Sets horizontal scroll for wide outputs
   #
   from IPython.display import display, HTML
   display(HTML(""))

   return 


#  # #############################################################


def f_init2():
    
   from katana import remote


   NUM_PARTITIONS  = 3
      #
   DB_NAME         = "my_db"
   GRAPH_NAME      = "my_graph"
   
    
   #  Get a client handle
   #
   my_client = remote.Client()
   
   #  Connect to graph
   #
   my_graph, *_ = my_client.get_database(name=DB_NAME).find_graphs_by_name(GRAPH_NAME)
       
    
   return my_client, my_graph


#  # #############################################################
#  # #############################################################


def f_init():
    
   f_init1()

   my_client, my_graph = f_init2()

    
   return (my_client, my_graph)


#  # #############################################################
#  # #############################################################


def f_get_token():
    
   import google.auth
   import google.auth.transport.requests
   from google.oauth2 import service_account


#  This token times out often; you must rerun this block from time to time
#
def f_get_token():
   l_credentials = service_account.Credentials.from_service_account_file(
      "/home/jovyan/work/My_KG_NoteBooks/P1_Prospects/13_DataBricks_Google/10_Data/05_katana-clusters-beta-d8605ac248e7.json",
      scopes=['https://www.googleapis.com/auth/cloud-platform'])
   l_auth_req = google.auth.transport.requests.Request()
   l_credentials.refresh(l_auth_req)
      #
   return l_credentials.token
     
    
l_token = f_get_token()

    
print("")
print("Token: " + l_token[0:120] + " ...")
print("")





















#  # #############################################################





