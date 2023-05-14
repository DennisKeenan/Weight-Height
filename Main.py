from pandas import *
from numpy import *
from seaborn import *
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split as train
from sklearn.linear_model import LinearRegression

# Print data head
data=read_csv("weight-height.csv")

# Edit data
data.rename(columns={"Height(Inches)":"Height","Weight(Pounds)":"Weight"},inplace=True)
data["Height"]=data["Height"]*2.54
data["Weight"]=data["Weight"]/2.205

# # Print data description
# print(data.head()) 
# print("="*30)
# print(data.describe())
# print("="*30)

# # Scatter Graphics
# plot.figure(figsize=(10,10))
# scatterplot(x=data["Height"],y=data["Weight"])

# # Histogram
# plot.figure(figsize=(7,6))
# histplot(data["Weight"])
# histplot(data["Height"])
# plot.show()

# Clearing nulls
print(data.isnull().sum())
data2=data.fillna(data.mean())
print(data2.isnull().sum())
data2=data2.drop("Gender",axis="columns")

# Data splitting
data_x=data2.drop("Weight",axis="columns")
data_y=data2.drop("Height",axis="columns")
print(data_x.shape)
print(data_y.shape)

# Training
x_train,x_test,y_train,y_test=train(data_x,data_y,test_size=0.2)
trainer=LinearRegression()
trainer.fit(x_train,y_train)
# print(trainer.coef_)
# print(trainer.intercept_)
print("="*30)

    # Predict 1 data
# height=float(input("Input your height: "))
# weight=trainer.coef_*height+trainer.intercept_
# print("Your weight is approximately: ",weight[0][0])

    # Predict dataset
y_ml=trainer.predict(x_test)
print(y_ml)
comp=DataFrame(c_[x_test,y_test,y_ml],columns=["Height","Weight","Weight_predict"])
print(comp.head(20))
scatterplot(data=[comp["Weight"],comp["Weight_predict"]])
plot.show()