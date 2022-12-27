
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


#  Google has a Web service to convert text into usable UMLS codes. See,
#        https://cloud.google.com/healthcare-api/docs/how-tos/nlp
#
#  In this cell, we begin to invoke this service on the text from the
#  cell above.

#  See also,
#     https://stackoverflow.com/questions/53472429/how-to-get-a-gcp-bearer-token-programmatically-with-python

#  Google:
#
#     .  We had to create an Auth Token, which produced a JSON file.
#        (Instruction in Url above.)
#
#     .  Our JSON file is at,
#              export GOOGLE_APPLICATION_CREDENTIALS="/mnt/hgfs/My.20/MyShare_1/46 Topics 2022/91 KG, All Prospects/13 KG, DataBricks, Google/10_Data/05_katana-clusters-beta-d8605ac248e7.json"
#              export GOOGLE_APPLICATION_CREDENTIALS="/home/jovyan/work/My_KG_NoteBooks/P1_Prospects/13_DataBricks_Google/10_Data/05_katana-clusters-beta-d8605ac248e7.json"
#
#     .  To extract the Auth Token, set the above, then run
#           gcloud auth application-default print-access-token


def f_get_token():
    
   import google.auth
   import google.auth.transport.requests
   from google.oauth2 import service_account

   l_credentials = service_account.Credentials.from_service_account_file(
      "/home/jovyan/work/My_KG_NoteBooks/P1_Prospects/13_DataBricks_Google/10_Data/05_katana-clusters-beta-d8605ac248e7.json",
      scopes=['https://www.googleapis.com/auth/cloud-platform'])
         #
   l_auth_req = google.auth.transport.requests.Request()
   l_credentials.refresh(l_auth_req)
      #
   l_token = l_credentials.token
 
   return l_token
     
    
def f_enrrich_int(i_arg1):
    
   import requests
   from requests.structures import CaseInsensitiveDict
      #
   import json


   l_url = "https://healthcare.googleapis.com/v1/projects/katana-clusters-beta/locations/us-central1/services/nlp:analyzeEntities"
   
   l_headers = CaseInsensitiveDict()
      #
   l_headers["Authorization"] = "Bearer " + l_token
   l_headers["Content-Type"]  = "application/json"
       
    
   l_data = """
      {{
      'nlpService':'projects/katana-clusters-beta/locations/us-central1/services/nlp',
      'documentContent':'{0}'
      }}
      """.format(i_arg1)
         #
   l_resp = requests.post(l_url, headers = l_headers, data = l_data)
      #
   return l_resp

















    
#  # #############################################################


def f_enrich():
    
   l_token = f_get_token()

   try:
      l_response    = f_enrich_int(l_each.transcription)
      l_data_asjson = json.loads(l_response.content) 
   except:
      l_data_asjson = None
        
   rerturn l_data_asjson







    
    
