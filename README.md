# hate-speech-classification
#### Language and Libraries

<p>
<a><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="python"/></a>
<a><img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/></a>
<a><img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/></a>
<a><img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="opencv"/></a>
<a><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/></a>
<a><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)" alt="docker"/></a>
<a><img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white" alt="gcp"/></a>
</p>


## Problem statement

In this project, we have created an API which predicts a wheather the user has given hateful comment or not. 

## Solution Proposed
The solution proposed for the above problem is that we have used NLP to solve the above problem.
We have used the Pytorch framework to solve the above problem also we used LSTM custom network with the help of PyTorch.
Then we created an API that takes in the text and classifies the type of feedback. Then we dockerized the application and deployed the model on the Google cloud.

## Dataset Used

There are two datasets one is imbalance data which consist of around 30000 rows and 2 column and the other dataset is the raw data which consist of 24000 rows and 7 columns. 

## How to run?

### Step 1: Clone the repository
```bash
git clone my repository 
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -p env python=3.8 -y
```

```bash
conda activate env
```

### Step 3 - Install the requirements
```bash
pip install -r requirements.txt
```

### Step 4 - Install Google Cloud Sdk and configure

#### For Windows
```bash
https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe
```
#### For Ubuntu
```bash
sudo apt-get install apt-transport-https ca-certificates gnupg
```
```bash
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
```
```bash
sudo apt-get update && sudo apt-get install google-cloud-cli
```
```bash
gcloud init
```
Before running server application make sure your `Google Cloud Storage` bucket is available

### Step 5 - Run the application server
```bash
python app.py
```

### Step 6. Train application
```bash
http://localhost:8080/docs
```

### Step 7. Prediction application
```bash
http://localhost:8080/docs
```

## Run locally

1. Check if the Dockerfile is available in the project directory

2. Build the Docker image

```
docker build -t hate-speech . 

```

3. Run the Docker image

```
docker run -d -p 8080:8080 <IMAGEID>
```

👨‍💻 Tech Stack Used
1. Python
2. FastAPI
3. Pytorch
4. Docker
5. NLP
6. LSTM

🌐 Infrastructure Required.
1. Google Cloud Storage
2. Google Compute Engine
3. Google Artifact Registry
4. Circle CI


## `hate` is the main package folder which contains 

**Artifact** : Stores all artifacts created from running the application

**Components** : Contains all components of Machine Learning Project
- DataIngestion
- DataTransformation
- ModelTrainer
- ModelEvaluation
- ModelPusher

**Custom Logger and Exceptions** are used in the project for better debugging purposes.


## Conclusion

- We have created a API which will classify the feedback is hateful or not.

=====================================================================