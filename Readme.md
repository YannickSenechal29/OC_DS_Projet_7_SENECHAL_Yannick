This repository contains all the work I have done for the project 7 of the OpenClassRooms DataScientist
training.
This project consist of making a scoring model of default payment for a loan company and deploy it on the web on an API.
A Dashboard made on streamlit is also available for customer service.
All codes part have been made in Python.
The data used for this project can be find on kaggle here: https://www.kaggle.com/c/home-credit-default-risk/data

Here is a summary of file and folder available on it:
- Folder API: Contain all the developpment work for the model API (code + csv file + pkl files + requirements).
- Folder Dahsboard: Contain all the developpement work for the dashboard (code + csv file + pkl files + png file + requirements).
- 3 Ipython jupyter notebook : These files are concerning all the work done with data from kaggle for selecting and training a  binary classifier model.
- Note_méthodologique.odt: French document to explain how we have selected, train and interpreted our scoring model
- P7_03_Support.odp: French presentation of the project

The API and the Dashboard have been deployed on Heroku and are available at these adresses:
- API: https://api-home-risk-oc-7.herokuapp.com/
- Dahsboard: https://dashboard-strl-home-risk-oc-7.herokuapp.com/

Please note that they also have specific github repository for deployment on Heroku:
- API: https://github.com/YannickSenechal29/API_home_risk.git
- Dahsboard: https://github.com/YannickSenechal29/Dashboard_home_risk.git 

Python librairies needed for the API are the following:
python = 3.8.11
pandas=1.3.3
numpy=1.20.3
scikit-learn=0.24.2
scipy=1.7.1
lightgbm=3.3.1
flask=2.0.2
gunicorn=20.1.0


Python librairies needed for the Dashboard are the following:
python = 3.8.11
pandas=1.3.3
numpy=1.20.3
scikit-learn=0.24.2
scipy=1.7.1
flask=2.0.2
seaborn=0.11.2
matplotlib=3.4.2
plotly=5.4.0
shap=0.40.0
streamlit=1.2.0
lightgbm=3.3.1
pillow=8.3.1

For further explainations and/or english translation for french documents don't hesitate to contact me at: ysenechal29@gmail.com

