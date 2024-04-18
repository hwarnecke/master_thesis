# master_thesis

Contains the code for my master thesis on the topic of LLM/RAG chatbots.



## Installation

I recommend creating a conda environment, as the requirement.txt only accounts for packages install with pip and not those that were already installed by conda. When using a different environment manager I (i.e. venv) I can't guarantee that there won't be some errors.

1. create a new conda  environment
   ``` shell
   conda create -n python=3.12 master_thesis
   ```

2. activate conda environment
   ``` shell
   conda activate master_thesis
   ```

3. install pip
   ``` shell
   conda install pip -y
   ```

4. install the requirements with pip
   ``` shell
   pip install -r requirements.txt
   ```

5. You also need to install some of the local python packages so that they can be imported from different folders. Currently there is only the on in data_handling.
   To install the package go into the folder with the setup.py file 
   ``` shell
   cd data_handling
   ```
   and run:
   ``` shell
   pip install .
   ```




## Folder Structure

The folder testing_features contains different scripts that test different features, mostly from the llama_index library. The files itself should have some comments explaining what I was testing and what I wanted to use that for, if you are interested.

The data_handling folder contains different scripts that are or were used to fetch data and create permanent llama_index compatible data stores, mostly using ElasticSearch.
When first cloning this repository you probably need to create the data store before running any RAG system. You can find the files for that in data_handling/DataStores/.
More specific instructions will follow when possible.



## ElasticSearch Docker

I use a local ElasticSearch instance running in a docker container. To start the container, navigate to the folder containing the docker-compose.yml file (mostly likely the root folder of the project). Maybe you need to install docker first.


If not already active, start the docker-daemon:
``` shell
sudo systemctl start docker
```
Then start the specific container:
``` shell
docker compose up city_services -d
```

The docker-compose.yml might specify different containers (i.e. different datasets or different embeddings). If you want to start a different one, replace ```city_service``` with the respective name of the container you want to start.

You can stop the instance from running with:
``` shell
docker compose down city_service
```
