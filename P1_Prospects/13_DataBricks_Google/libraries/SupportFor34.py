
#  Push the boilerplate code to a supporting file
#


#  # #############################################################
#  # #############################################################


#  Display options
#
def f_display_int():
    
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
def f_connect_int():
    
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
    
   f_display_int()

   my_client, my_graph = f_connect_int()

    
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
#  # #############################################################


def f_ready_for_graph_int(i_arg1, i_arg2):
    
   import pandas as pd


   df_PatientVisit        = pd.DataFrame([
      [ i_arg2   , str(i_arg1)           , "PatientVisit"],
         #
      [ "XX-1001", "{'x-col': 'XX-1001'}", "PatientVisit"],
      [ "XX-1002", "{'x-col': 'XX-1002'}", "PatientVisit"],
      [ "XX-1003", "{'x-col': 'XX-1003'}", "PatientVisit"],
         #
      ], columns = ["id", "transcription", "LABEL"])
         #
   df_UmlsEntityNodes     = pd.DataFrame([
      ["XX-1001", "XX-1001", "Unknown", "UmlsEntity"],
      ["XX-1002", "XX-1002", "Unknown", "UmlsEntity"],
      ["XX-1003", "XX-1003", "Unknown", "UmlsEntity"],
      ["XX-1004", "XX-1004", "Unknown", "UmlsEntity"],
         #
      ], columns = ["id", "entity_id", "preferred_term", "LABEL"])
         #
   df_UmlsVocabularyNodes =  pd.DataFrame([
#     ["XX-1001", "UmlsVocabulary"],
#     ["XX-1002", "UmlsVocabulary"],
#     ["XX-1003", "UmlsVocabulary"],
#     ["XX-1004", "UmlsVocabulary"],
         #
      ], columns = ["id", "LABEL"])


   df_PatientVisitToEntityEdge_N = pd.DataFrame([
      ["XX-1001", "XX-1004", "VISIT_CONTAINS"],
      ["XX-1002", "XX-1003", "VISIT_CONTAINS"],
      ["XX-1003", "XX-1002", "VISIT_CONTAINS"],
      ["XX-1004", "XX-1001", "VISIT_CONTAINS"],
         #
      ], columns = ["start_id", "end_id", "TYPE"])
         #
   df_PatientVisitToEntityEdge_S = pd.DataFrame([
      ["XX-1001", "XX-1004", "VISIT_CONTAINS"],
      ["XX-1002", "XX-1003", "VISIT_CONTAINS"],
      ["XX-1003", "XX-1002", "VISIT_CONTAINS"],
      ["XX-1004", "XX-1001", "VISIT_CONTAINS"],
         #
      ], columns = ["start_id", "end_id", "TYPE"])
         #
   df_EntityToVocabularyEdge_N   = pd.DataFrame([
      ["XX-1001", "XX-1004", "VISIT_CONTAINS"],
      ["XX-1002", "XX-1003", "VISIT_CONTAINS"],
      ["XX-1003", "XX-1002", "VISIT_CONTAINS"],
      ["XX-1004", "XX-1001", "VISIT_CONTAINS"],
         #
      ], columns = ["start_id", "end_id", "TYPE"])
         #
   df_EntityToVocabularyEdge_S   = pd.DataFrame([
      ["XX-1001", "XX-1004", "VISIT_CONTAINS"],
      ["XX-1002", "XX-1003", "VISIT_CONTAINS"],
      ["XX-1003", "XX-1002", "VISIT_CONTAINS"],
      ["XX-1004", "XX-1001", "VISIT_CONTAINS"],
         #
      ], columns = ["start_id", "end_id", "TYPE"])
         
    
#  What we get from our test data,
#
#     {'entityMentions': [
         {
         'mentionId': '1',
         'type': 'PROBLEM',
         'text': {'content': 'mole', 'beginOffset': 4},
         'linkedEntities': [
            {'entityId': 'UMLS/C0027960'},
            {'entityId': 'UMLS/C0027962'}
            ],
         'temporalAssessment': {'value': 'UPCOMING', 'confidence': 0.6084825992584229},
         'certaintyAssessment': {'value': 'LIKELY', 'confidence': 0.057481713593006134},
         'subject': {'value': 'PATIENT', 'confidence': 0.9589220881462097},
         'confidence': 0.5596652626991272
         },
         {
         'mentionId': '2',
         'type': 'ANATOMICAL_STRUCTURE',
         'text': {'content': 'ear', 'beginOffset': 15},
         'linkedEntities': [
            {'entityId': 'UMLS/C0013443'}
            ],
     'confidence': 0.9025713205337524}],
     'entities': [{'entityId': 'UMLS/C0013443',
     'preferredTerm': 'Ear structure',
     'vocabularyCodes': ['FMA/52780',
     'FMA/77739',
     'LNC/LA21929-7',
     'LNC/LA22163-2',
     'LNC/LP7188-8',
     'LNC/MTHU001407',
     'MSH/D004423',
     'MTH/NOCODE',
     'NCI/C12394',
     'OMIM/MTHU000052']},
     {'entityId': 'UMLS/C0027960',
     'preferredTerm': 'Nevus',
     'vocabularyCodes': ['HPO/HP:0003764',
     'MEDLINEPLUS/4491',
     'MEDLINEPLUS/5402',
     'MSH/D009506',
     'MTH/NOCODE',
     'OMIM/MTHU016869']},
     {'entityId': 'UMLS/C0027962',
     'preferredTerm': 'Melanocytic nevus',
     'vocabularyCodes': ['HPO/HP:0000995',
     'HPO/HP:0003764',
     'MSH/D009508',
     'MTH/NOCODE',
     'NCI/C7570']}]}

    
   #  "entityMentions" should be a root level key to this dictionary
   #

   if ("entityMentions" in i_arg1):
      #
      #  Loop thru these
      #
      for l_each in i_arg1["entityMentions"]:
         if ("linkedEntities" in l_each):
            for l_entity in l_each["linkedEntities"]:
               if ("entityId" in l_entity):
                  print("AAA")
                  #
                  #  Build a dictionary that we will append to the DataFrame
                  #
                  l_recd1 = { "id": [l_entity["entityId"]], "entity_id" : [l_entity["entityId"]], "LABEL": ["UmlsEntity"] }
                  #
                  #  If this key is present, add it to the dictionary
                  #
                  if ("preferredTerm" in l_each):
                     print("BBB")
                     #
                     #  We have an additional key, add to the record 
                     #
                     l_recd1.update( {"preferred_term": [str(l_entity["preferredTerm"])]} )
                  else:
                     print("CCC")
                     l_recd1.update( {"preferred_term": ["Unknown"                     ]} )
                        #
                  df_UmlsEntityNodes = df_UmlsEntityNodes.append( pd.DataFrame(l_recd1) )
                
                
                
                  #
                  #  Above was our list of Nodes of LABEL "UmlsEntity"
                  #  
                  #  Here we make our Edge list from;  PatientVisit --> UmlsEntity
                  #
                  #  We make all Edges to be bi-directional. As a heterogeneous relationship,
                  #  we need two arrays.
                  #
                  l_recd2a = { "start_id": i_arg2                   , "end_id": str(l_entity["entityId"]), "TYPE": "VISIT_CONTAINS" }
                  l_recd2b = { "start_id": str(l_entity["entityId"]), "end_id": i_arg2                   , "TYPE": "VISIT_CONTAINS" }
                     #
                  df_PatientVisitToEntityEdge_N.append(l_recd2a, ignore_index = True)
                  df_PatientVisitToEntityEdge_S.append(l_recd2b, ignore_index = True)
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
                        #  l_recd3 = { "id": l_vocab, "vocabulary_code": l_vocab, "LABEL": "UmlsVocabulary" }
                        l_recd3 = { "id": l_vocab, "vocabulary_code": "MMM", "LABEL": "UmlsVocabulary" }
                           #
                        df_UmlsVocabularyNodes.append(l_recd3, ignore_index = True)
                        #
                        #  And create the Edge from UmlsEntity --> UmlsVocabulary
                        #
                        l_recd4a = { "start_id": str(l_entity["entityId"]), "end_id": str(l_vocab             ), "TYPE": "ALSO_CODED_AS" }
                        l_recd4b = { "start_id": str(l_vocab             ), "end_id": str(l_entity["entityId"]), "TYPE": "ALSO_CODED_AS" }
                           #
                        df_EntityToVocabularyEdge_N.append(l_recd4a, ignore_index = True)
                        df_EntityToVocabularyEdge_S.append(l_recd4b, ignore_index = True)
               else:
                  print("GGG")
                    
    
   return  df_PatientVisit, df_UmlsEntityNodes, df_UmlsVocabularyNodes, df_PatientVisitToEntityEdge_N, df_PatientVisitToEntityEdge_S, df_EntityToVocabularyEdge_N, df_EntityToVocabularyEdge_S
        
        
#  # #############################################################


def f_insert_into_graph(i_arg1, i_arg2, i_arg3, i_arg4, i_arg5, i_arg6, i_arg7, i_arg8):
    
   from katana.remote import import_data

   with import_data.DataFrameImporter(i_arg1) as df_importer:   
        
      #  Just nodes
      #
      df_importer.nodes_dataframe(
         i_arg2,
         id_column             = "id",
         id_space              = "PatientVisit",  
         label                 = "PatientVisit",  
         ) 
      df_importer.nodes_dataframe(
         i_arg3,
         id_column             = "id",
         id_space              = "UmlsEntity",  
         label                 = "UmlsEntity",  
         ) 
      df_importer.nodes_dataframe(
         i_arg4,
         id_column             = "id",
         id_space              = "UmlsVocabulary",  
         label                 = "UmlsVocabulary",  
         ) 

      #  Just edges
      #
#     df_importer.edges_dataframe(
#        i_arg5, 
#        source_id_space       = "PatientVisit", 
#        destination_id_space  = "UmlsEntity",   
#        source_column         = "start_id",
#        destination_column    = "end_id",
#        type                  = "VISIT_CONTAINS"
#        )
#     df_importer.edges_dataframe(
#        i_arg6, 
#        source_id_space       = "UmlsEntity", 
#        destination_id_space  = "PatientVisit",   
#        source_column         = "start_id",
#        destination_column    = "end_id",
#        type                  = "VISIT_CONTAINS"
#        )
#     df_importer.edges_dataframe(
#        i_arg7, 
#        source_id_space       = "UmlsEntity", 
#        destination_id_space  = "UmlsVocabulary",   
#        source_column         = "start_id",
#        destination_column    = "end_id",
#        type                  = "ALSO_CODED_AS"
#        )
#     df_importer.edges_dataframe(
#        i_arg8, 
#        source_id_space       = "UmlsVocabulary", 
#        destination_id_space  = "UmlsEntity",   
#        source_column         = "start_id",
#        destination_column    = "end_id",
#        type                  = "ALSO_CODED_AS"
#        )

      df_importer.insert()
    
    
   return
        
        
#  # #############################################################
#  # #############################################################


from random import randint
   #
l_uniqkey = randint(2000,2999)


def f_enrich(i_arg1, i_arg2):
    
   import json
      #
   global l_uniqkey


   l_token = f_get_token_int()
      #
   try:
      l_response    = f_enrich_int(i_arg1, l_token)
      l_data_asjson = json.loads(l_response.content) 
   except:
      l_data_asjson = None
        
        
   l_uniqkey += 1
   l_uniqid  = str("PV-" + str(l_uniqkey))
      #
   l_df1, l_df2, l_df3, l_df4, l_df5, l_df6, l_df7 = f_ready_for_graph_int(l_data_asjson, l_uniqid)
      #
   #  f_insert_into_graph(i_arg2, l_df1, l_df2, l_df3, l_df4, l_df5, l_df6, l_df7)


   print(l_df2)
   print("TTT")
   print(l_df3)
        
   return l_data_asjson, l_uniqid