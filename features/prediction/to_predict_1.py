import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df=pd.read_csv(r'~/Projects/MID/mid/features/data/TEST_TEST.csv')
Labelencoder=LabelEncoder()
df.Attrition=Labelencoder.fit_transform(df.Attrition)
data = df.select_dtypes(include=[np.number]).interpolate().dropna()
x=df.drop("Attrition",axis=1)
y=df["Attrition"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf = RandomForestClassifier(n_estimators=500)
model = clf.fit(x_train, y_train)

joblib.dump(model,"to_predict_1.pkl")
