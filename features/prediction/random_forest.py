import pandas as pd
import numpy  as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
import joblib 

attrition = pd.read_csv(r"~/Projects/MID/mid/features/data/dataset.csv")
# Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
attrition["Attrition_numerical"] = attrition["Attrition"].apply(lambda x: target_map[x])

# Refining our list of numerical variables
numerical = [u'Age', u'DailyRate',  u'JobSatisfaction',
       u'MonthlyIncome', u'PerformanceRating',
        u'WorkLifeBalance', u'YearsAtCompany', u'Attrition_numerical']

# Drop the Attrition_numerical column from attrition dataset first - Don't want to include that
attrition = attrition.drop(['Attrition_numerical'], axis=1)

# Empty list to store columns with categorical data
categorical = []
for col, value in attrition.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attrition.columns.difference(categorical)

# Store the categorical data in a dataframe called attrition_cat
attrition_cat = attrition[categorical]
attrition_cat = attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column

attrition_cat = pd.get_dummies(attrition_cat)

# Store the numerical features to a dataframe attrition_num
attrition_num = attrition[numerical]

# Concat the two dataframes together columnwise
attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)

# Define a dictionary for the target mapping
# Use the pandas apply method to numerically encode our attrition target variable
target_map = {'Yes':1, 'No':0}
target = attrition["Attrition"].apply(lambda x: target_map[x])

train, test, target_train, target_val = train_test_split(attrition_final, target,train_size= 0.80,random_state=0)
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_sample(train,target_train)

seed = 0   # We set our random seed to zero for reproducibility
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
#   'warm_start': True, 
    'max_features': 0.3,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}

rf = RandomForestClassifier(**rf_params)
rf.fit(smote_train, smote_target)
rf_predictions = rf.predict(test)

print("Accuracy score: {}".format(accuracy_score(target_val, rf_predictions)))
print("="*80)

joblib.dump(rf, 'random_forest.pkl') 

def fit_for_dummy(attr):
    attr_dummies = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 'EmployeeNumber',
     'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
     'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
     'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
     'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
     'YearsWithCurrManager', 'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
     'Department_Human Resources', 'Department_Research & Development', 'Department_Sales', 'EducationField_Human Resources', 'EducationField_Life Sciences',
     'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Female', 'Gender_Male',
     'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
     'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
     'JobRole_Sales Representative', 'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single', 'Over18_Y', 'OverTime_No', 'OverTime_Yes']
    
    for i in range(55) :
        attr_dummies[i] = -1

    numerical = [u'Age', u'DailyRate', u'DistanceFromHome', u'Education',u'EmployeeCount', u'EmployeeNumber',
            u'EnvironmentSatisfaction', u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction', 
            u'MonthlyIncome',u'MonthlyRate',u'NumCompaniesWorked',u'PercentSalaryHike',u'PerformanceRating',
            u'RelationshipSatisfaction',u'StandardHours', u'StockOptionLevel', u'TotalWorkingYears',u'TrainingTimesLastYear',
            u'WorkLifeBalance', u'YearsAtCompany',u'YearsInCurrentRole',u'YearsSinceLastPromotion',u'YearsWithCurrManager']

    categorical=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']

    attr_dummies = attr[numerical]

    if attr['BusinessTravel'] == 'Non-Travel' :
        attr_dummies['BusinessTravel_Non-Travel'] = 1
        attr_dummies['BusinessTravel_Travel_Frequently'] = 0
        attr_dummies['BusinessTravel_Travel_Rarely'] = 0
    elif attr['BusinessTravel'] == 'Travel_Frequently' :
        attr_dummies['BusinessTravel_Non-Travel'] = 0
        attr_dummies['BusinessTravel_Travel_Frequently'] = 1
        attr_dummies['BusinessTravel_Travel_Rarely'] = 0
    else:
        attr_dummies['BusinessTravel_Non-Travel'] = 0
        attr_dummies['BusinessTravel_Travel_Frequently'] = 0
        attr_dummies['BusinessTravel_Travel_Rarely'] = 1



    if attr['Department'] == 'Human Resources':
        attr_dummies['Department_Human Resources'] = 1
        attr_dummies['Department_Research & Development'] = 0
        attr_dummies['Department_Sales'] = 0
    elif attr['Department'] == 'Research & Development':
        attr_dummies['Department_Human Resources'] = 0
        attr_dummies['Department_Research & Development'] = 1
        attr_dummies['Department_Sales'] = 0
    elif attr['Department'] == 'Sales'  :
        attr_dummies['Department_Human Resources'] = 0
        attr_dummies['Department_Research & Development'] = 0
        attr_dummies['Department_Sales'] = 1

# 'EducationField_Human Resources', 'EducationField_Life Sciences',
# 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other',
# 'EducationField_Technical Degree'

    if attr['EducationField'] == 'Human Resources':
        attr_dummies['EducationField_Human Resources'] = 1
        attr_dummies['EducationField_Life Sciences'] = 0
        attr_dummies['EducationField_Marketing'] = 0
        attr_dummies['EducationField_Medical'] = 0
        attr_dummies['EducationField_Other'] = 0
        attr_dummies['EducationField_Technical Degree'] = 0
    elif attr['EducationField'] == 'Life Sciences':
        attr_dummies['EducationField_Human Resources'] = 0
        attr_dummies['EducationField_Life Sciences'] = 1
        attr_dummies['EducationField_Marketing'] = 0
        attr_dummies['EducationField_Medical'] = 0
        attr_dummies['EducationField_Other'] = 0
        attr_dummies['EducationField_Technical Degree'] = 0
    elif attr['EducationField'] == 'Marketing':
        attr_dummies['EducationField_Human Resources'] = 0
        attr_dummies['EducationField_Life Sciences'] = 0
        attr_dummies['EducationField_Marketing'] = 1
        attr_dummies['EducationField_Medical'] = 0
        attr_dummies['EducationField_Other'] = 0
        attr_dummies['EducationField_Technical Degree'] = 0
    elif attr['EducationField'] == 'Medical':
        attr_dummies['EducationField_Human Resources'] = 0
        attr_dummies['EducationField_Life Sciences'] = 0
        attr_dummies['EducationField_Marketing'] = 0
        attr_dummies['EducationField_Medical'] = 1
        attr_dummies['EducationField_Other'] = 0
        attr_dummies['EducationField_Technical Degree'] = 0
    elif attr['EducationField'] == 'Techical Degree':
        attr_dummies['EducationField_Human Resources'] = 0
        attr_dummies['EducationField_Life Sciences'] = 1
        attr_dummies['EducationField_Marketing'] = 0
        attr_dummies['EducationField_Medical'] = 0
        attr_dummies['EducationField_Other'] = 0
        attr_dummies['EducationField_Technical Degree'] = 1
    else:
        attr_dummies['EducationField_Human Resources'] = 0
        attr_dummies['EducationField_Life Sciences'] = 1
        attr_dummies['EducationField_Marketing'] = 0
        attr_dummies['EducationField_Medical'] = 0
        attr_dummies['EducationField_Other'] = 1
        attr_dummies['EducationField_Technical Degree'] = 0

    if attr['Gender'] == 'Male':
        attr_dummies['Gender_Male'] = 1
        attr_dummies['Gender_Female'] = 0
    else:
        attr_dummies['Gender_Male'] = 0
        attr_dummies['Gender_Female'] = 1

# 'JobRole_Healthcare Representative', 'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
# 'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
# 'JobRole_Sales Representative'
    
    if attr['JobRole'] == 'Healthcare Representative':
        attr_dummies['JobRole_Healthcare Representative'] = 1
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0

    elif attr['JobRole'] == 'Human Resources':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 1
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0
    
    elif attr['JobRole'] == 'Laboratory Technician':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 1
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0

    elif attr['JobRole'] == 'Manager':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 1
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0
    
    elif attr['JobRole'] == 'Manufacturing Director':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 1
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0

    elif attr['JobRole'] == 'Research Director':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 1
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0

    elif attr['JobRole'] == 'Research Scientist':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 1
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 0

    elif attr['JobRole'] == 'Sales Executive':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 1
        attr_dummies['JobRole_Sales Representative'] = 0

    elif attr['JobRole'] == 'Sales Representative':
        attr_dummies['JobRole_Healthcare Representative'] = 0
        attr_dummies['JobRole_Human Resources'] = 0
        attr_dummies['JobRole_Laboratory Technician'] = 0
        attr_dummies['JobRole_Manager'] = 0
        attr_dummies['JobRole_Manufacturing Director'] = 0
        attr_dummies['JobRole_Research Director'] = 0
        attr_dummies['JobRole_Research Scientist'] = 0
        attr_dummies['JobRole_Sales Executive'] = 0
        attr_dummies['JobRole_Sales Representative'] = 1
    
# 'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single',   

    if attr['MaritalStatus'] == 'Divorced':
        attr_dummies['MaritalStatus_Divorced'] = 1
        attr_dummies['MaritalStatus_Married'] = 0
        attr_dummies['MaritalStatus_Single'] = 0
    elif attr['MaritalStatus'] == 'Married' :
        attr_dummies['MaritalStatus_Divorced'] = 0
        attr_dummies['MaritalStatus_Married'] = 1
        attr_dummies['MaritalStatus_Single'] = 0
    else:
        attr_dummies['MaritalStatus_Divorced'] = 0
        attr_dummies['MaritalStatus_Married'] = 0
        attr_dummies['MaritalStatus_Single'] = 1

# 'Over18_Y'  

    attr_dummies['Over18_Y'] = 1

#  'OverTime_No', 'OverTime_Yes'
    if attr['OverTime'] == 'No':
        attr_dummies['OverTime_No'] = 1
        attr_dummies['OverTime_Yes'] = 0
    else:
        attr_dummies['OverTime_No'] = 0
        attr_dummies['OverTime_Yes'] = 1        

    print(type(attr_dummies))
    return attr_dummies

test_array = attrition.iloc[14,:]
data_series = fit_for_dummy(test_array)

model = joblib.load("random_forest.pkl")

data_array = np.array(data_series)
data_list = data_array.reshape(1,-1)
print(model.predict(data_list))

print(data_series)