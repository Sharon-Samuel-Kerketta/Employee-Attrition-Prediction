import csv
import os
fields = [] 
rows = [] 
with open(os.path.join('features/data/',"TEST_TEST.csv")) as csvfile: 
    csvreader = csv.reader(csvfile) 
    # extracting field names through first row 
    fields = next(csvreader) 
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 

def convertlist(datalist):
    datadic=dict()
    datadic={
        "Age": datalist[0],
        "Attrition" : datalist[1],
        "DistanceFromHome" : datalist[2],
        "MonthlyIncome" : datalist[3],
        "NumCompaniesWorked" : datalist[4],
        "PercentSalaryHike" : datalist[5],
        "StandardHours" : datalist[6],
        "StockOptionLevel" : datalist[7],
        "TotalWorkingYears" : datalist[8],
        "TrainingTimesLastYear" : datalist[9],
        "YearsAtCompany" : datalist[10],
        "YearsSinceLastPromotion" : datalist[11],
        "YearsWithCurrManager" : datalist[12]
    }
    return datadic

def fetch_data_csv():
    new_rows=[]
    for row in rows : 
        datadic = convertlist(row)
        new_rows.append(datadic)
    return new_rows