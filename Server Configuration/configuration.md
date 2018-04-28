## Use GPU

### Install GPU Driver
http://www.nvidia.com/download/driverResults.aspx/132256/en-us

### Install CUDA 9.0
+ Verify GPU is CUDA compatible: ```lspci | grep -i nvidia```
+ Install gcc: ```apt install gcc```
+ ```wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb```
+ ```sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb```
+ ```sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub```
+ ```sudo apt-get update```
+ ```sudo apt-get install cuda-9-0```

### Install cuDNN
+ Download cuDNN 7.0 from https://developer.nvidia.com/rdp/cudnn-download
+ Upload to Server: ```sudo gcloud compute scp libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb root@nlp-server-01:~/installers/.```; ```sudo gcloud compute scp libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb root@nlp-server-02:~/installers/.```
+ Install: ```sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb```

### For Theano
+ Add ```export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64``` to ```.bashrc```
+ Add ```export CUDA_ROOT=/usr/local/cuda``` to ```.bashrc```

## Install miniconda
+ download: ```wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh```
+ make runnable: ```chmod +x ...```
+ install: ```./...sh```
+ ```source .bashrc```

## Miniconda environment
+ ```conda create --name py2 python=2```
+ ```conda install pip```
+ ```conda install ipython```

## Theano install
+ ```pip install numpy scipy mkl```
+ ```pip install Theano==0.9.0```

### Tensorflow install
+ ```pip install --upgrade tensorflow-gpu```