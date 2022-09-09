#!/bin/bash


USE_PORT=7777


#  Tested on; KatanaGraph Client VM release "focal", 20.04.2
#
#     cat /etc/os-release


echo ""
echo "This program will perform all steps necessary to boot a MongoDB database"
echo "(version 5.0.7) that will operate on localhost:${USE_PORT}"
echo ""
echo "   .  The expectation is that you would run this program from within the"
echo "      [ local ] KatanaGraph Client VM, and that any Jupyter notebook would"
echo "      interact with KatanaGraph and MongoDB both, acting as an intermediary."
echo ""
echo "   .  The MongoDB database will run in the background (forked). You will"
echo "      be given the PID."
echo ""
echo "   .  Any data placed in the MongoDB database will be preserved between"
echo "      MongoDB restarts."
echo ""
echo ""
echo "**  You have 10 seconds to cancel before proceeding."
echo ""
echo ""
sleep 10


###################################################


echo ""
echo "Checking (apt) and related dependencies ..."
echo ""
apt update                                                                   &> /dev/null
apt -y install curl                                                          &> /dev/null
apt -y install iproute2                                                      &> /dev/null
   #
apt-get -y install libcurl4 libgssapi-krb5-2 libldap-2.4-2 libwrap0 libsasl2-2 libsasl2-modules libsasl2-modules-gssapi-mit snmp openssl liblzma5  &> /dev/null
   #
apt --fix-broken install                                                     &> /dev/null
   #
apt update                                                                   &> /dev/null

pip install pymongo                                                          &> /dev/null


###################################################


[ ! -d "05_MDB" ] && {
   mkdir 05_MDB
   mkdir 05_MDB/data
   chmod -R 777 05_MDB
   }

cd 05_MDB
   #
[ ! -f "bin/mongo" ] && {
   echo ""
   echo "MongoDB binary not found: installing now ..."
   echo ""
      #
   curl https://downloads.mongodb.com/linux/mongodb-linux-x86_64-enterprise-ubuntu2004-5.0.7.tgz -o mongo.tgz  &> /dev/null
   tar xf mongo.tgz   &> /dev/null
      #
   rm -r mongo.tgz
      #
   cd mongodb-linux-x86_64-enterprise-ubuntu2004-5.0.7
   mv * ..
   cd ..
   rm -r mongodb-linux-x86_64-enterprise-ubuntu2004-5.0.7
   }


###################################################


echo ""
echo "Starting MongoDB ..."
echo ""
./bin/mongod --dbpath ./data --logpath ./logfile --bind_ip 0.0.0.0  --port ${USE_PORT} --fork  &> /dev/null
l_pid=`ps -ef | grep "./bin/mongod" | head -1 | awk '{print $2}'`


echo "The MongoDB database server PID is: "${l_pid}
echo ""

echo ""
echo "Next Steps:"
echo ""
echo "   .  You can't run the MongoDB 'mongo' command line client locally"
echo "      because of the echo mode (lack of raw I/O) on the Client VM."
echo ""
echo "   .  In a Jupyter Notebook, you can run, (indentation purposely suppressed)"
echo ""

echo '''

import pymongo
   # 
from pymongo import MongoClient

   ###

'''

echo "cn = MongoClient(\"localhost:${USE_PORT}\")"

echo '''
db = cn.my_database

db.my_collection.drop()

db.my_collection.insert_one( {"state" : "XX", "zip" : 55555 } )
db.my_collection.insert_one( {"state" : "CO", "zip" : 66666 } )
db.my_collection.insert_one( {"state" : "WI", "zip" : 77777 } )
db.my_collection.insert_one( {"state" : "TX", "zip" : 88888 } )
db.my_collection.insert_one( {"state" : "YY", "zip" : 99999 } )
   
   ###
   
print("Executing a find()")
print("")
sss = list ( db.my_collection.find( { "state" : { "$in" : [ "CO" , "WI" ] } } ) )
for s in sss:
   print(s)

db.my_collection.count_documents({})
'''


echo ""
echo ""





