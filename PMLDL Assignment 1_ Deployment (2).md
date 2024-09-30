# PMLDL Assignment 1: Deployment

This assignment concerns the topic of MLOps and consists of the **main task** and **extra task** for bonus points. In the main task, you are asked to **deploy a machine learning model**. Additionally, as an extra task, you can elaborate on the deployment part and create a **simple automated MLOps pipeline** with the tools that you learnt in the lab.

Solving the main task is enough to get a full grade for this assignment. Meanwhile, solving the extra task will provide additional 5 bonus points to the course grade. The result of your assignment should be a **GitHub repository** with the code implementing the tasks.

# Main Task

In this task you need to **deploy a model in an API** and **create a web application** that interacts with the API. The model API accepts the requests from the web application and sends back the responses. The web application must at least contain the input fields, button for make a prediction, and the area with the prediction itself. To do that, you are supposed to use Docker containers, FastAPI and Streamlit frameworks. __Implementing data and model engineering pipelines is not required for this task__. Your task is only to deploy the model API and the web application.

## Recommended Steps to Complete the Main Task

Here we provide the steps to deploy your model. However, you can do it in different ways, just make sure that your repository is structured (has the hierarchy of files of folders that is logical) and works.


1. Create a GitHub repository. Make sure that your repository is __public__. Clone the repository. The directory of the cloned repository is now your working directory.
2. Write some Python code to train a machine learning model. Save the code in the `code/models` folder. Save the file with trained model in the `models` folder.
3. Write a Python script that implements the model API using FastAPi. Write a Dockerfile for the API. Store both of these files in the `code/deployment/api` folder.
4. Write a Python script that implements the web application using Streamlit. Write a Dockerfile for the application. Store both of these files in the `code/deployment/app` folder.
5. Write a docker-compose file that includes the API and the web application. Store the docker-compose file in the `code/deployment` folder.
6. Build and run docker-compose.
7. Commit the files and push them to GitHub.

## Expected Repository Structure

Here is the structure of the repository that you are encouraged to follow:

```yaml
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api
│   │   └── app
│   └── models
├── data
└── models
```

## Other Notes

* You are allowed to use any model and dataset, except Iris dataset, which was presented in the lab. Keep in mind that using Iris dataset will lead to **deduction of 50% of the points gained**.
* You are also allowed to use the frameworks other than FastAPI and Streamlit. Nevertheless, the usage of Docker is a must-have. Ignoring the usage of Docker will lead to **0 points** for the assignment.
* If the your trained model is too large for GitHub, then you are allowed to not to push the model to GitHub

## Submission

Submit the solution of the task as a **link to the GitHub repository**. After the deadline of the link submission there will be arranged a **meeting with TAs** where you have to show that your deployment works.

## Main Task Grading Criteria

* **Model API container** works correctly - 2 points
* **Web application container** works correctly - 2 points
* The repository is **structured** (has the hierarchy of files of folders that is logical) - 1 point

## Useful Links for Main Task

* [Docker Tutorial for Beginners](https://docker-curriculum.com/)
* [FastAPI Website](https://fastapi.tiangolo.com/)
* [Get Started with Streamlit](https://docs.streamlit.io/get-started)


# Extra Task

In this task, you are asked to compose a **simple automated pipeline** with the tools that you learnt in the lab. The pipeline should contain the three stages: **1) data engineering**, **2) model engineering**, and **3) deployment**. The automation of the pipeline should be implemented as **automatic pipeline running each 5 minutes** (if the pipeline takes more time to run, you can increase the period of time between pipeline runs).  

## Stages Description

## Stage 1: Data Engineering

### Input Artifacts

* File(s) with raw data

### Output Artifacts

* File with training data
* File with testing data

### Operations

This stage involves **data loading**, **data cleaning**, and **data splitting**. Firstly, the pipeline loads the data by reading it from the file(s). Then, the pipeline should clean the data by removing/imputing missing values and removing the outliers. At the end of this stage, the pipeline should split the data into train and test datasets and save them in the corresponding files.

### Tools

The operations of the Stage 1 may be implemented using data pipelines of **DVC** or using **Airflow** tasks.

### Additional Notes

As the dataset you can use any data that you want. The size of the dataset is also up to you.

## Stage 2: Model Engineering

### Input Artifacts

* File with training data
* File with testing data

### Output Artifacts

* File with a trained model
* Values of the testing metrics

### Operations

This stage involves **feature engineering**, **model** **training, evaluation,** and **packaging**. When the pipeline starts this stage, it should obtain the training and testing data from the previous stage. The pipeline firstly runs feature engineering routines to transform training and testing data into the features for the model. Then, a model is trained using the features of the training data. At the end, the trained model should be evaluated by calculating performance metrics on the testing features. The testing metrics should be logged. The trained model should be saved (packaged) in a file.

### Tools

This stage may be entirely performed in **MLflow**.

### Additional Notes

In this stage it is enough to train a single model. However, you can train multiple models if you want. The type and complexity of the model is up to you. Even a simple model will be enough for this stage. You can also do hyperparameter tuning, but it is not necessary. The choice of the extension of the file with the trained model is up to you.

## Stage 3: Deployment

### Input Artifacts

* File with a trained model

### Output Artifacts

* Running model API
* Running app

### Operations

This stage involves **model API deployment** and **app deployment**. The pipeline should create a Docker image with model API and run it in a container. Also, the pipeline should create a Docker image with the app communicating with the model API. The application should contain the input fields, button to make the prediction. After pressing the button the predictions of the model should be shown.

### Tools

Use Docker to run the API and the app! For the API you can use FastAPI or other web framework. For the app you can use Streamlit or similar frameworks.

### Additional Notes

It is strongly recommended to use Docker for model and app deployment. Ignoring this requirement will lead to the grade decrease. Also, the API and the app should be deployed in different containers. Ignoring this will also decrease your grade. The app may be very simple and contain the input fields to enter input data, button to run prediction, and the prediction itself. 

## Recommended Structure of Your Repository

Here is the structure of the repository that you are encouraged to follow:

```yaml
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api
│   │   └── app
│   └── models
├── data
│   ├── processed
│   └── raw
├── notebooks
├── models
├── services
│   └── airflow
│       ├── dags
│       └── logs
└── requirements.txt
```

## Particular Steps to Implement the Pipeline

Here we provide a more concrete steps to implement the pipeline. You can follow them if you have a lack of understanding what to do.


 1. Create a GitHub repository. Make sure that your repository is __public__. Clone the repository. The directory of the cloned repository is now your working directory.
 2. Create a new virtual environment in your working directory.
 3. Create a `requirements.txt` file with the list of Python libraries necessary for your pipeline. Always keep this file updated with a fresh list of Python libraries.
 4. Download the data that you will use for model training and validation. Create the `data/raw` folder and place the files with the data to this folder.
 5. Write code to load, clean, and split the data using DVC or Airflow. Make the code save the splitted data in `data/processed` folder. If you are using Airflow, save the files with the code in `serivces/airflow/dags`, otherwise save the code to `code/datasets` folder.
 6. Write code to create features for the model. Implement model training, evaluation, and packaging. Log the metrics and the model in MLflow. Save the code to `code/models` folder. Save the trained model in `models` directory. 
 7. Implement API for the model using FastAPI. Save the code to `code/deployment/api` folder. Write Dockerfile for the API and save it in `code/deployment/api/Dockerfile`
 8. Implement the application using Streamlit. Save the code to `code/deployment/app` folder. Write Dockerfile for the application and save it in `code/deployment/app/Dockerfile`
 9. Write a docker-compose file and save it in `code/deployment/docker-compose.yml` 
10. Construct a pipeline performing all the stages. The pipeline should run the code from the steps 5, 6 and 9. Save the code of the pipeline in `serivces/airflow/dags`. Make sure that the pipeline works and set the schedule for 5 minutes.

## Other Notes

* You can use other tools than suggested. You can also add another tools to the pipeline, but  the specified stages should be present in the pipeline.
* Any dataset and models could be used. Small and simple models could be used: the important thing is the pipeline.

## Submission

Submit the solution of the assignment as a **link to the GitHub repository**. After  the deadline of the link submission there will be arranged a **meeting with TAs** where you have to show that your pipeline is working.

## Extra Task Grading Criteria

 The grade for the assignment is calculated as the sum of the points for the following criteria:

* **Data processing** stage is implemented and working - 1 point
* **Model engineering** stage implemented and working - 1 point
* **Deployment** stage implemented and working - 1 point
* The pipeline is entirely **automated** - 1 point
* The repository is **structured** (has the hierarchy of files of folders that is logical) - 1 point

## Useful Links for Extra Task

* [DVC Get Started Guide](https://dvc.org/doc/start)
* [MLFlow Get Started Guide](https://mlflow.org/docs/latest/getting-started/index.html)
* [FastAPI Website](https://fastapi.tiangolo.com/)
* [Docker Tutorial for Beginners](https://docker-curriculum.com/)
* [Apache Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/1.10.15/tutorial.html)


* [Get Started with Streamlit](https://docs.streamlit.io/get-started)


