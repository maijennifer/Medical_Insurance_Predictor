# Medical-Insurance-Prediction

Mai Le, Okechukwu Odunze(OC), Meera Patel, Elaine Kellerman

## Website URL https://medical-insurance-prediction.streamlit.app/

## Project Overview 

In this project, we built a Pandas Database using Collab, Machine Learning, Xgboost as well as other dictionary methods to extract and transform data. We accessed the insurance ipynb files to find out the accurate prediction details about the charges of Medical insurance costs based off an individual’s age, sex, BMI, Children, Smoker status, Region and used the filtered Dataset to create graphs and data lists to show the result.

## Dataset

Our dataset is used to predict insurance charges based on the person’s age, sex, body mass index, number of children, smoking status, and region. We used the dataset to understand and analyze which variables can affect a person’s insurance premiums. Age is likely to be the most important variable as young customers and old customers may pay more or less. BMI is another important element when predicting insurance cost due to different factors like obesity risks and likely to affect premium pricing decisions. Because smokers are more likely to develop health concerns and therefore represent a great risk, insurance companies may charge up to 50% more for premiums. Lastly, number of children and location has big effect on customers premiums. Differences in competition, local and state policies, and cost of living account for the location charges. 

## Machine Learning Model

XGBoost is one of the most popular supervised learning machine models. This extreme grade boosting machine uses the decision tree ensemble, training each subset to each tree to combine  to a final prediction model. As each model in succession correct the errors of the previous on a graded scale, this allows XGboost machine learning to give you the best linear regression for your data set with lower risk of overfitting.
For our machine model, XGboost loops through multiple settings of 6 different parameters setting to perform almost 13,000 fittings to find combination of parameters for the best accurate machine. With the best fitted parameters shown below, our machine model was able to predict 83% of testing data and 89 % of training data, with RMSE of 0.06239, all for under 10 minutes.

Predicting healthcare cost for individual using accurate prediction models is important. Knowing ahead the estimated cost could allow customers to choose insurance plans with appropriate deductibles and premiums.  We analyzed all the elements influencing pricing decisions: age, BMI, sex, smoking status, children, location; we also examined different predictive models that could calculate an appropriate estimation.  Using our machine model can help individuals and families better predict the cost of health care insurance. 



Datasets Used:

Visual Studio Code,
Machine Learning,
Jupyter Notebook,

Libraries:

GridScaler,
Xgboost,
Stremlit,
Pandas,
Sklearn,
Min/Max Scaler,
Joblib,
Pathlib.

Work Cited: Wakefield, B.  Prediction of Insurance Charges. A Study of Customer’s Insurance Charges. https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender
