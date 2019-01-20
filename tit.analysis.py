
# coding: utf-8

# In[138]:

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[139]:

train_dat=pd.read_csv("/Users/StevenTseng/Desktop/titanic.dataset/train.csv")
test_dat=pd.read_csv("/Users/StevenTseng/Desktop/titanic.dataset/test.csv")
all=[train_dat,test_dat]


# In[140]:

#Factorial Variables
train_dat.describe(include=["O"])


# In[141]:

#Quantitive Variables
train_dat.describe()


# In[80]:

train_dat[["Pclass","Survived"]].groupby("Pclass",as_index=False).mean()


# In[79]:

train_dat[["Sex","Survived"]].groupby("Sex",as_index=False).mean()


# In[84]:

train_dat[["Parch","Survived"]].groupby("Parch",as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[89]:

g = sns.FacetGrid(train_dat, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()


# In[93]:

grid = sns.FacetGrid(train_dat, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();plt.show()


# In[98]:

grid = sns.FacetGrid(train_dat, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare')
grid.add_legend();plt.show()


# In[142]:

train_dat = train_dat.drop(['Ticket', 'Cabin'], axis=1)
test_dat = test_dat.drop(['Ticket', 'Cabin'], axis=1)


# In[143]:

train_dat["Title"]=train_dat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_dat["Title"]=test_dat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[144]:

train_dat["Title"]=train_dat["Title"].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],"Rare")
train_dat["Title"]=train_dat['Title'].replace(["Mme","Mlle","Ms"],"Miss")


# In[145]:

test_dat["Title"]=test_dat["Title"].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],"Rare")
test_dat["Title"]=test_dat['Title'].replace(["Mme","Mlle","Ms"],"Miss")


# In[146]:

pd.crosstab(test_dat['Title'],test_dat["Sex"])


# In[149]:

all=[train_dat,test_dat]


# In[150]:

#mappping the Title

title_mapping={'Mr': 1, 'Mrs':3, 'Miss':2, 'Master':4, 'Rare':5}
for dataset in all:
    dataset['Title']= dataset['Title'].map(title_mapping)
    dataset["Title"]=dataset["Title"].fillna(0)


# In[151]:

all[0]=all[0].drop(["Name","PassengerId"],axis=1)
all[1]=all[1].drop("Name",axis=1)


# In[152]:

#mappping the Sex
sex_mapping={"male":0,"female":1}
for dataset in all:
    dataset["Sex"]=dataset["Sex"].map(sex_mapping).astype(int)


# In[157]:

#Missing Value

all[0].apply(lambda x : sum(x.isnull()),axis=0)
[all[0].shape,all[1].shape]


# In[158]:

#Age

guess_ages = np.zeros((2,3))
guess_ages



# In[161]:

for dataset in all:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# In[162]:

train_dat["AgeBand"]=pd.cut(train_dat["Age"],5)
train_dat[["AgeBand","Survived"]].groupby(by="AgeBand").mean()


# In[163]:

for dataset in all:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
all[0].head()


# In[164]:

for dataset in all:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


all[0][["FamilySize","Survived"]].groupby(by="FamilySize").mean().sort_values(by="Survived",ascending=False)


# In[166]:

# Variable : "Alone"
for dataset in all:
    dataset["Isalone"]=0
    dataset.loc[dataset["FamilySize"]==1,"Isalone"]=1


# In[167]:

#Fill values in Embarked

for dataset in all:
    dataset['Embarked'] = dataset['Embarked'].fillna("S")
    

all[0][["Embarked","Survived"]].groupby(by="Embarked").mean()


# In[170]:

for dataset in all:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

all[0].head()


# In[171]:

#Fill values in Fare


all[1]["Fare"].fillna(all[1]["Fare"].median(),inplace=True)


# In[172]:

for dataset in all:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)



# In[175]:

[all[0].columns,all[1].columns]


# In[177]:

x_train=all[0].drop("Survived",axis=1)
y_train=all[0]["Survived"]
x_test=all[1].drop("PassengerId",axis=1)

[x_train.shape,y_train.shape,x_test.shape]


# In[231]:

#LogisticRegression

Logreg=LogisticRegression()

Logreg.fit(x_train,y_train)

Y_pred=Logreg.predict(x_test)

acc_log = round(Logreg.score(x_train, y_train) * 100, 2)


# In[232]:

corr_df=pd.DataFrame({"Feature":all[0].columns.delete(0)})

corr_df["Coef"]=pd.Series(Logreg.coef_[0])

corr_df.sort_values(by="Coef",ascending=False)


# In[233]:

#Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_test)
acc_rf = round(random_forest.score(x_train, y_train) * 100, 2)

acc_rf


# In[237]:

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree



# In[235]:

models = pd.DataFrame(
{"Model":["Logis","Random Forest","Decision Tree"],
"Score":[acc_log,acc_rf,acc_decision_tree]
         }
)

models


# In[238]:

submission = pd.DataFrame({
        "PassengerId": all[1]["PassengerId"],
        "Survived": Y_pred
    })


# In[240]:

submission.to_csv("/Users/StevenTseng/Desktop/titanic.dataset/submission.csv",index=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

6


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[35]:

all


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



