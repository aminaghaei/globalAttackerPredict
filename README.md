# Global Terrorist Attack Predictor
This repository contains one MatLab code and one Python code for solving the "Global Terrorist Attacks" challenge of startup.ml using machine learning models. The MatLab and Python codes are independent and each can be used seprately to solve the problem. The python code is presented in a Jupyter notebook format.

In the Matlab code, logistic regression with nonlinear decision boundaries is used. The user can go up to third order for mapping the features to the higher order terms. No machine learning library is used in this code. A sigmoid-based cost function is evaluated for each classifier and then minimized. The first order, second order and third order models predict 47%, 67% and 76% of the cross validation data correctly, respectively.

In the Python code, the Scikit-learn library is used. Two different classifiers (linear logistic regression and random forest) are used to demonstrate the advantage of one over the other for this data set. With the linear regression model we get recall and precision of 0.46 and 0.43 respectively which is close to what the MatLab code predicts using linear logistic regression. With the random forest model we get 0.93 for both precision and recall.

---------------------------------------------------------------------------------------

Here is the challenge: https://startup.ml/challenge

Global Terrorism Database (GTD) is an open-source database including information on terrorist events around the world from 1970 through 2014. Some portion of the attacks have not been attributed to a particular terrorist group.

Use attack type, weapons used, description of the attack, etc. to build a model that can predict what group may have been responsible for an incident.
