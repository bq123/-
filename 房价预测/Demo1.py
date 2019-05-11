import numpy as np
import  pandas as  pd
data =pd.read_csv('house_price.csv')
data1=data.dropna()

data2=pd.get_dummies(data1[['dist','floor']])
pd.set_option('display.max_columns',None)
data3=data2.drop(['dist_shijingshan','floor_high'],axis=1)#去掉shijingshan和high
data4=pd.concat([data3,data1[['roomnum','halls','AREA','subway','school','price']]],axis=1)#合并data1和data3
#print(data4)
x=data4.iloc[:,:-1]
y=data4.iloc[:,-1:]
#print(y)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,test_size=0.3,random_state=42)#%30为实验 %70 为预测
model=linear_model.LinearRegression().fit(x_train,y_train)
result=model.predict(np.array([[0,0,0,0,0,0,0,2,1,60,1,1]]))#设置参数
print(result)#  打印预测值
print(model.coef_)#模型系数
print(model.intercept_)#模型截距
print(model.score(x_text,y_text))