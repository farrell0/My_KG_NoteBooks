

#  This file is a stand alone Python executable.
#
#  More,
#
#     .  Run this program as;  python C00_simple_client.py
#        "client", as in "client/server".
#
#        This program runs on the Jupyter Pod of a Katana cluster operating
#        atop Kubernetes.
#
#        In total, this set of programs require the following dependencies,
#           pip install flask flask_graphql graphene extraction requests
#
#        Again; this would be installed on the Jupyter pod.
#
#     .  The call below to requests() sends an Http request to whatever is listening
#        at localhost, port 5000.
#
#        When we run another program in this set;  python P00_web_server.py
#        there will be an agent listening at localhost, port 5000.
#        This second program is our "server".
#
#           See that file for more details on what is operating, and what that
#           agent provides.
#
#     .  The data we send (l_query) is a [ JSON like ] string formatted as a GraphQL
#        query.
#
#        A Jupyter NoteBook in this project creates the graph, creates data, other.
#
#        Notice that the graph was created with (airport_code, airport_name, ..),
#        but the GraphQL query uses camel case.
#


import requests


l_query = """
{ 
  airport(airportCode: "SJC") {
    airportCode
    airportName
    LABEL
  }
}
"""

l_result = requests.post("http://localhost:5000/", params = {"query": l_query})
   #
print()
print(l_result.json())
print()




