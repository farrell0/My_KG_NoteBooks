

#  Script used to install TensorFlow o nthe KD Cluster worker nodes ..

sudo mkdir -p /tmp/tensorboard
sudo mkdir -p /tmp/models
sudo chmod 777 /tmp/tensorboard
sudo chmod 777 /tmp/models
 
sudo /usr/local/bin/pip install tensorboard
sudo /usr/local/bin/pip install setuptools==59.5.0
sudo /usr/local/bin/pip install testresources