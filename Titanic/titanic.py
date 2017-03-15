import pandas as pd #analytics
import numpy as np #sci computation
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv("/home/heather/myGithub/Kaggle/01Titanic/train.csv")

'''
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"Survival Situation(1 as survived)")
plt.ylabel(u"Number of People")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"Number of people")
plt.title(u"Passenger Class Distribution")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"Age")
plt.title(u"distribution based on age(1 as survived)")
plt.grid(b=True,which="major",axis='y')

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind="kde")
data_train.Age[data_train.Pclass==2].plot(kind="kde")
data_train.Age[data_train.Pclass==3].plot(kind="kde")
plt.xlabel(u"Age")
plt.ylabel(u"Density")
plt.title(u"Age distribution of passenger class")
plt.legend((u"1st class",u"2nd class",u"3rd class"),loc="best")

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"#passenger embarked at each place")
plt.ylabel(u"Number of people")
'''

# class yes
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u"survived":Survived_1, u"not survived":Survived_0})
df.plot(kind="bar",stacked=True)
plt.title(u"survival situation of different class")

#sex yes
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u"male":Survived_m, u"female":Survived_f})
df.plot(kind="bar",stacked=True)
plt.show()