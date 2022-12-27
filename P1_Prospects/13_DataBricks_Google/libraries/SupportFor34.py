
#  Push the boilerplate code to a supporting file
#


#  # #############################################################
#  # #############################################################


#  Display options
#
def f_display():
    
   import pandas as pd
   from tabulate import tabulate


   pd.set_option("display.width", 480)
   
   #  Sets horizontal scroll for wide outputs
   #
   from IPython.display import display, HTML
   display(HTML(""))

   return 


#  # #############################################################


#  Connect to the KG server
#
def f_connect():
    
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
    
   f_display()

   my_client, my_graph = f_connect()

    
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


def f_get_token_int():
    
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
     
    
def f_enrich_int(i_arg1, i_arg2):
    
   import requests
   from requests.structures import CaseInsensitiveDict
      #
   import json


   l_url = "https://healthcare.googleapis.com/v1/projects/katana-clusters-beta/locations/us-central1/services/nlp:analyzeEntities"
   
   l_headers = CaseInsensitiveDict()
      #
   l_headers["Authorization"] = "Bearer " + i_arg2
   l_headers["Content-Type"]  = "application/json"
       
    
   l_data = """
      {{
      'nlpService':'projects/katana-clusters-beta/locations/us-central1/services/nlp',
      'documentContent':'{0}'
      }}
      """.format(i_arg1)
         #
   l_response = requests.post(l_url, headers = l_headers, data = l_data)
      #
   return l_response

    
#  # #############################################################


def f_add_to_graph(i_arg1):
    
    
    
    
    
    
    
    
    
    
   #  "entities" should be a root level key to this dictionary
   #
   if ("entities" in i_arg1):
      #
      #  Loop thru these
      #
      for l_entity in l_each_asdict["entities"]:
            
            l_cntr += 1
               #
            if (l_cntr % 100000 == 0):
               print("")
               print("Processed so far: %d" % (l_cntr))
            else:
               if (l_cntr % 1000 == 0):
                  print(".", end = "")
            
            if ("entityId" in l_entity):
               #
               #  Build a dictionary that we will append to an array
               #
               l_recd1 = { "id": l_entity["entityId"], "entity_id" : l_entity["entityId"], "LABEL": "UmlsEntity" }
               #
               #  If this key is present, add it to the dictionary
               #
               if ("preferredTerm" in l_entity):
                  #
                  #  We have an additional key, add to the record and add to our array
                  #
                  l_recd1.update( {"preferred_term": l_entity["preferredTerm"]} )
                     #
               l_UmlsEntityNodes.append(l_recd1)
               #
               #  Above was our list of Nodes of LABEL "UmlsEntity"
               #  
               #  Here we make our Edge list from;  PatientVisit --> UmlsEntity
               #
               #  We make all Edges to be bi-directional. As a heterogeneous relationship,
               #  we need two arrays.
               #
               l_recd2a = { "start_id": str(l_each.id)           , "end_id":   str(l_entity["entityId"]), "TYPE": "VISIT_CONTAINS" }
               l_recd2b = { "start_id": str(l_entity["entityId"]), "end_id":   str(l_each.id)           , "TYPE": "VISIT_CONTAINS" }
                  #
               l_PatientVisitToEntityEdge_N.append(l_recd2a)
               l_PatientVisitToEntityEdge_S.append(l_recd2b)
               #
               #  We are done with UmlsEntity and its Edge to PatientVisit
               #
               #  Also in "entities" is another array, "vocabularyCodes"
               #
               if ("vocabularyCodes" in l_entity):
                  for l_vocab in l_entity["vocabularyCodes"]:
                     #
                     #  Add to our set of Vocabulary Nodes
                     #
                     l_recd3 = { "id": l_vocab, "vocabularyCode": l_vocab, "LABEL": "UmlsVocabulary" }
                        #
                     l_UmlsVocabularyNodes.append(l_recd3)
                     #
                     #  And create the Edge from UmlsEntity --> UmlsVocabulary
                     #
                     l_recd4a = { "start_id": str(l_entity["entityId"]), "end_id": str(l_vocab             ), "TYPE": "ALSO_CODED_AS" }
                     l_recd4b = { "start_id": str(l_vocab             ), "end_id": str(l_entity["entityId"]), "TYPE": "ALSO_CODED_AS" }
                        #
                     l_EntityToVocabularyEdge_N.append(l_recd4a)
                     l_EntityToVocabularyEdge_S.append(l_recd4b)






















#  # #############################################################



def f_enrich(i_arg1):
    
   import json


   l_token = f_get_token_int()

   try:
      l_response    = f_enrich_int(i_arg1, l_token)
      l_data_asjson = json.loads(l_response.content) 
   except:
      l_data_asjson = None
        
   return l_data_asjson


#  # #############################################################
#  # #############################################################









    
    
