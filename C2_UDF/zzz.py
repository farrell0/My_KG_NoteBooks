
from katana import remote

my_client = remote.Client()

print(my_client)


for t in my_client.operations():
   print(t)





