import pandas as pd #analytics
import numpy as np #sci computation
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("train.csv")


fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind = "bar")
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


# class correlation:yes------------------>
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u"survived":Survived_1, u"not survived":Survived_0})
df.plot(kind="bar",stacked=True)
plt.title(u"survival situation of different class")

#sex correlation: yes------------------>
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u"male":Survived_m, u"female":Survived_f})
df.plot(kind="bar",stacked=True)
plt.show()

 #survival condition based on sex under different Pcalss------------------>
fig=plt.figure()
fig.set(alpha=0.65)
plt.title(u"survival condition based on sex under different Pcalss")

ax1=fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"Survive", u"not survived"], rotation=0)
ax1.legend([u"female/first-class"], loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"not survived", u"Survive"], rotation=0)
plt.legend([u"female/lowest-class"], loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"not survived", u"Survive"], rotation=0)
plt.legend([u"male/first-class"], loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"not survived", u"Survive"], rotation=0)
plt.legend([u"male/lowest-class"], loc='best')
plt.show()


#Explore siblings correlation" not really------------------>
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print df

g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print df


# feature engineering------------------>
# Lost-Data filling
def set_missing_ages(df):

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y: target age
    y = known_age[:, 0]

    # X: feature value
    X = known_age[:, 1:]

    # fit to RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # predict
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # fill data
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


#to be continued
