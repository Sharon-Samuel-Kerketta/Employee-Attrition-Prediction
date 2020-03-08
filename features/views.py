from django.shortcuts import render
from features.prediction.rfor import calculate_feature_importances
from features.prediction.data_pro import predict_attr
from features.data.read_csv import fetch_data_csv
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


import random
import pandas as pd
import joblib

def index(request):
    return render(request,'index.html')

def index_table(request):
    rows=fetch_data_csv()
    random.shuffle(rows)
    rows = rows[:25][:]
    context={
        "rows": rows
    }
    return render(request,'index_table.html',context)

@csrf_exempt
def prediction(request):
    context={}
    age=request.POST.get('Age')
    distance=request.POST.get('DistanceFromHome')
    income=request.POST.get('MonthlyIncome')
    worked=request.POST.get('NumCompaniesWorked')
    hike=request.POST.get('PercentSalaryHike')
    hours=request.POST.get('StandardHours')
    stock=request.POST.get('StockOptionLevel')
    years=request.POST.get('TotalWorkingYears')
    training=request.POST.get('TrainingTimesLastYear')
    years_company=request.POST.get('YearsAtCompany')
    promo=request.POST.get('YearsSinceLastPromotion')
    manager=request.POST.get('YearsWithCurrManager')

    data = [age, distance,income,worked,hike,hours,stock,years,training,years_company,promo,manager]
    data_for_template = {'age' : age, "distance" : distance, "income" : income, "worked" : worked,"hike" : hike,"hours":hours,"stock":stock,"years":years,"training":training,"years_company":years_company,"promo":promo,"manager":manager}
    
    inp_df = pd.DataFrame([data])
    model = joblib.load('./features/prediction/to_predict_1.pkl')
    values = model.predict_proba([inp_df])
    attrition = model.predict([inp_df])
    
    if attrition[0] == 1:
        attrition = int(1)
    else:
        attrition = int(0)
    satisfaction_rate = values[0][0] * 100

    context = {
        'attrition' : attrition,
        'values' : satisfaction_rate,
        'emp_data' : data_for_template
    }
    return render(request,'precision.html',context)
    # return JsonResponse(context)

def feature_importances(request):

    values=calculate_feature_importances()
    context=dict()
    context={
        'values': values
    }
    return render(request,'feature_importances.html',context)

@csrf_exempt
def predict_attr_fn(request): 
    Age=request.POST.get('Age')
    DailyRate=request.POST.get('DailyRate')
    DistanceFromHome=request.POST.get('DistanceFromHome')
    Education=request.POST.get('Education')
    EmployeeCount=request.POST.get('EmployeeCount')
    EmployeeNumber=request.POST.get('EmployeeNumber')
    EnvironmentSatisfaction=request.POST.get('EnvironmentSatisfaction')
    HourlyRate=request.POST.get('HourlyRate')
    JobInvolvement=request.POST.get('JobInvolvement')
    JobLevel=request.POST.get('JobLevel')
    JobSatisfaction=request.POST.get('JobSatisfaction')
    MonthlyIncome=request.POST.get('MonthlyIncome')
    MonthlyRate=request.POST.get('MonthlyRate')
    NumCompaniesWorked=request.POST.get('NumCompaniesWorked')
    PercentSalaryHike=request.POST.get('PercentSalaryHike')
    PerformanceRating=request.POST.get('PerformanceRating')
    RelationshipSatisfaction=request.POST.get('RelationshipSatisfaction')
    StandardHours=request.POST.get('StandardHours')
    StockOptionLevel=request.POST.get('StockOptionLevel')
    TotalWorkingYears=request.POST.get('TotalWorkingYears')
    TrainingTimesLastYear=request.POST.get('TrainingTimesLastYear')
    WorkLifeBalance=request.POST.get('WorkLifeBalance')
    YearsAtCompany=request.POST.get('YearsAtCompany')
    YearsInCurrentRole=request.POST.get('YearsInCurrentRole')
    YearsSinceLastPromotion=request.POST.get('YearsSinceLastPromotion')
    YearsWithCurrManager=request.POST.get('YearsWithCurrManager')
    BusinessTravel=request.POST.get('BusinessTravel')
    Department=request.POST.get('Department')
    EducationField=request.POST.get('EducationField')
    Gender=request.POST.get('Gender')
    JobRole=request.POST.get('JobRole')
    MaritalStatus=request.POST.get('MaritalStatus')
    Over18=request.POST.get('Over18')
    MaritalStatus=request.POST.get('MaritalStatus')
    OverTime =request.POST.get('OverTime')

    data_for_template = {'Age' : Age, "DailyRate":DailyRate, "DistanceFromHome" : DistanceFromHome,
    "Education":Education,"EmployeeCount":EmployeeCount,"EmployeeNumber":EmployeeNumber,
    "EnvironmentSatisfaction":EnvironmentSatisfaction,"HourlyRate":HourlyRate,"JobInvolvement":JobInvolvement,
    "JobLevel":JobLevel,"JobSatisfaction":JobSatisfaction,"MonthlyIncome" : MonthlyIncome,
    "MonthlyRate" : MonthlyRate,"NumCompaniesWorked" : NumCompaniesWorked,"PercentSalaryHike":PercentSalaryHike,
    "PerformanceRating":PerformanceRating,"RelationshipSatisfaction":RelationshipSatisfaction,
    "StandardHours":StandardHours,"StockOptionLevel":StockOptionLevel,"TotalWorkingYears":TotalWorkingYears,
    "TrainingTimesLastYear":TrainingTimesLastYear,"WorkLifeBalance":WorkLifeBalance,"YearsAtCompany":YearsAtCompany,
    "YearsInCurrentRole":YearsInCurrentRole,"YearsSinceLastPromotion":YearsSinceLastPromotion,
    "YearsWithCurrManager":YearsWithCurrManager,"BusinessTravel":BusinessTravel,"Department":Department,
    "EducationField":EducationField,"Gender":Gender,"JobRole":JobRole,"MaritalStatus":MaritalStatus,
    "Over18":Over18," MaritalStatus": MaritalStatus,"OverTime":OverTime}

    inp_df = pd.Series(data_for_template)
    value = predict_attr(inp_df)
    data = {"data" : data_for_template,
        "attrition_values" : value}

    return JsonResponse(data)

