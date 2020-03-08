import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def calculate_feature_importances():
    df=pd.read_csv(os.path.join('features/data','TEST_TEST.csv'))

    Labelencoder=LabelEncoder()
    df.Attrition=Labelencoder.fit_transform(df.Attrition)

    # the model can only handle numeric values so filter out the rest
    data = df.select_dtypes(include=[np.number]).interpolate().dropna()

    x=df.drop("Attrition",axis=1)
    y=df["Attrition"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(x_train, y_train)

    values = sorted(zip(x_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
    
    return values