

#  Generally, this library file provides our GraphQL 'resolvers'.
#
#  Resolvers are basically,
#
#     .  The methods assigned to each of the expected CRUD operations.
#        Below we support methods for retrieve of airport nodes via primary
#        key, and new node creation.
#
#     .  A proof of technology, we supplied simple methods only-
#        You could, of course, supply end points for any OpenCyhper
#        traversal, Katana AI routine, other.
#
#     .  What GraphQL (Graphene) provides is a data driven uery langauge.
#        Minimallyy then, GraphQL provides,
#
#        .. Under fetching (reducing columns)
#        .. Over fetching  (combining SELECTS/MATCHes) using UNION
#        .. (Other)
#
#     .  Graphene gives us our GraphQL server, including, should we wish,
#        a built in Web query UI.
#


####################################################


#  L01_katana gives us our Katana graph conenction handle
#

import graphene
import pandas as pd
   #
from L01_katana import my_graph


####################################################


#  Both methods below are just Python helper functions, simple
#  data access objects (DAOs)
#
#  Recall; our graph uses property names equal to (airport_code,
#  airport_name, ..) where GraphQL wants camel case.
#

def get_airport(airport_code):
    
   l_result = my_graph.query_unpaginated("""

      MATCH (n: Airport) 
      WHERE n.airport_code = '{0}'
      RETURN n.airport_code AS airportCode, n.airport_name AS airportName, LABELS(n)[0] AS LABEL
      
      """.format(airport_code) )
         #
    
   l_return = l_result    # [0].to_dict()
    
    
   return l_return


def add_airport(airport_code, airport_name, LABEL):
    
   l_query = """
   
      CREATE ( n: {0} {{ airport_code: '{1}' }} )   
      SET n.airport_name = '{2}'
      
      """.format(LABEL, airport_code, airport_name)

   l_result = pd.DataFrame(my_graph.query_unpaginated(l_query))
         #
   return True


####################################################


#  A class for our primary data object, an Airport
#

class Airport(graphene.ObjectType):
   airportCode  = graphene.String(required = True)
   airportName  = graphene.String()
   LABEL        = graphene.String()


####################################################
####################################################


class Query(graphene.ObjectType):

   #  'airport' becomes a keyword here, referenced in any
   #  GraphQL Query string, [ and ] in the 'resolve_'
   #  method below ..

   airport = graphene.Field(Airport, airportCode = graphene.String(), airportName = graphene.String(), LABEL = graphene.String())
    
   def resolve_airport(self, info, airportCode):
      #  l_result =  pd.DataFrame( get_airport(airportCode) )
      l_result =  get_airport(airportCode) 
      print(type(l_result))
         #
        
      return l_result
        
#     return Airport(
#        airportCode = l_result.airportCode,
#        airportName = l_result.airportName,
#        LABEL       = l_result.LABEL,
#        )


   ###
    

#  class CreateAirport(graphene.Mutation):
#      
#     class Arguments:
#        airportCode = graphene.String()
#        airportName = graphene.String()
#        LABEL       = graphene.String()
#  
#     airport = graphene.Field(lambda: Airport)
#  
#     def mutate(root, info, airportCode, airportName, LABEL):
#           #
#        add_airport(airportCode, airportName, LABEL)
#        airport = Airport(airportCode = airportCode, airportName = airportName, LABEL = LABEL)
#           #
#        return CreateAirport(airport = airport)
#  
#  
#  class Mutate(graphene.ObjectType):
#      create_airport = CreateAirport.Field()
    
    
####################################################


#  Surface all of the contents of this file for use by our Python Flask
#  Web server
#
#  l_schema = graphene.Schema(query = Query, mutation = Mutate, types = [Airport])
l_schema = graphene.Schema(query = Query, types = [Airport])





