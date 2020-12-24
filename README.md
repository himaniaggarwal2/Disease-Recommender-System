# Disease-recommender System
This is a Disease Recommender System that I have created with the minimalist dataset taken from Kaggle.Here we had 3 datasets namely for disease symptoms, precautions and description of the respective diseases.We created a final dataset named mySddf by removing all duplicates and containing all the features required for working with the given dataset and the model we want to create. Then processing of the dataset is done on the basis of 
* NLP (Natural Language Processing)
* SVC(Support Vector Machine)
* OneVsRest
* Scikit Learn

[![GitHub top language](https://img.shields.io/github/languages/top/himaniaggarwal2/disease-recommender?color=green&logo=python)](https://github.com/himaniaggarwal2/disease-recommender) [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/himaniaggarwal2/disease-recommender?logo=github)](https://github.com/himaniaggarwal2/disease-recommender) [![GitHub commit activity](https://img.shields.io/github/commit-activity/m/himaniaggarwal2/disease-recommender?color=bluevoilet&logo=github)](https://github.com/himaniaggarwal2/disease-recommender/commits/) [![GitHub repo size](https://img.shields.io/github/repo-size/himaniaggarwal2/disease-recommender?logo=github)](https://github.com/himaniaggarwal2/disease-recommender)


---
## Getting Started

1. Fork [this](https://github.com/himaniaggarwal2/disease-recommender.git) repository.
2. Clone this forked repository in your local system by the following commands:
> git clone https://github.com/__your-github-username__/disease-recommender.git

You can find the above link by opening this in your github account and copy the url that is being showed in the Address bar u will notice the two in the system.


```
Note: The following commands will be working on Unix,Linux and MacOS
```
3. To Navigate to the project directory.
   >ls

   >cd  disease-recommender

4. Create a new branch in it.
   > git checkout -b `<your-branch-name>`
5. To check whether you are not in master branch 
   >git branch

check the star in front of the two
   
```
    master
    *<your-branch-name>
```
5. Make changes in it.
6. Run it.
7. Stage your changes and commit it.

You need to maintain the version control of the changes u made. It helps when working on the current repository.

Add changes to index

>git add .

Commit to the local repository

>git commit -m "<your_commit_message>"

8. Push your local commits to your local repository.
>git push -u origin <your_branch_name>
----
## Pre-Requisites 
Pre-Requisites to run this project on your System are 

* Python 3

* Anaconda 

* Jupyter Notebook Installed in your system

    Jupyter notebook through your terminal is lighter option for your system.

(MAC OS)
```
brew install jupyter

pip3 install numpy

pip3 install pandas

pip3 install matplotlib

pip3 install streamlit
```
---
## About this Repository
* ```Final Dataset.ipynb``` file stores the cleaning and feature engineering upon our datasets curated from our top 3 datasets.
* ```chatbot_recommender.ipynb``` stored the model prepared and worked done.
* ```healthbot.py``` creates streamlit web application that can be deployed from your local server.

----
# Deploying Web Application:
* Run the below command on your server
> streamlit run healthbot.py 

---
This project took a lot of hardwork so do give it a star and fork it in your system to know more about it ;)



[![built with love](https://forthebadge.com/images/badges/built-with-love.svg)](https://www.linkedin.com/in/himaniaggarwal2/) [![smile please](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://github.com/himaniaggarwal2/)



