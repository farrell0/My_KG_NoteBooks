

#  This file is a stand alone Python executable.
#
#  More,
#
#     .  Run this program as;  python P00_web_server.py
#        "server", as in "client/server".
#
#        This program will stand up an Http server at localhost and
#        port 5000, both default values. Run in the foreground, this program
#        will never terminate. A CONTROL-C can be used to terminate this
#        program.
#
#        This program runs on the Jupyter pod of a Katana cluster operating
#        atop Kubernetes.
#
#        In total, this set of programs require the following dependencies,
#           pip install flask flask_graphql graphene extraction requests
#
#     .  How you access this server-
#
#        ..  You can run the 00_simple_client.py program in this same
#            folder.
#
#        ..  Because of the parameter below (graphiql = True), you can also
#            open a Web browser to localhost:5000, and interact with a Web UI.
#
#            This Web server will be operating from the Jupyter pod of a Katana
#            cluster operating atop Kubernetes.
#
#            How does your local laptop Web browser access said Web server ?
#            We can use kubectl port forwarding.
#            So, for example,
#
#            ...  Your kubectl can access your pods.
#                 Your Jupyter pod is discovered to be named, (for example),
#                    std-8-katana-notebook-farrell
#                 and resides in the Kubernetes namespace titled,
#                    product-sandbox
#          
#                 You would then run a,
#                    kubectl port-forward std-8-katana-notebook-farrell -n product-sandbox 5000:5000
#
#                 Whatever is running on that pod, at port 5000, is now accessible via your
#                 laptop at port 5000.
#

from flask import Flask
   #
from flask_graphql import GraphQLView


from L01_katana import my_graph
from L02_schema import l_schema


####################################################


my_app = Flask(__name__)

my_app.add_url_rule(
   "/",
   view_func=GraphQLView.as_view(
      "graphql",
      schema   = l_schema,
      graphiql = True,
      context  = {"session": my_graph}
      )
   )

if __name__ == "__main__":
   my_app.run()







