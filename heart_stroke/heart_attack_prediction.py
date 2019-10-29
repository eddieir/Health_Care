import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./input"))

train_data= pd.read_csv("./input/train_2v.csv")
test_data=pd.read_csv("./input/test_2v.csv")

test_data=pd.read_csv("./input/test_2v.csv")

print(train_data.head())

print(train_data.info())

print("Train_Data Set Shape : {}" .format(train_data.shape))

print("Test_Data Set Shape : {}" .format(test_data.shape))

#Train_Data Set Shape : (43400, 12)
#Test_Data Set Shape : (18601, 11)

print(train_data.describe())

'''
Data preprocessing
'''

((train_data.isnull().sum()/len(train_data))*100).sort_values(ascending=False)


#Cleary bioth the data set has missing value for smoking Status and Bmi. Lets join bith the data set

joined_data=pd.concat([train_data,test_data])
print("joined Data Shape : {}" .format(joined_data.shape))
joined_data.iloc[-5:-1]
a=((joined_data.isnull().sum()/len(joined_data))*100).sort_values(ascending=False)

a.plot.bar()
plt.savefig("./output/figure1.png")




#lets impute BMI with its Mean as it has very less missing records

train_data['bmi']=train_data.bmi.fillna(train_data.bmi.mean())

((train_data.isnull().sum()/len(train_data))*100).sort_values(ascending=False)

joined_data=pd.concat([train_data,test_data])

print("joined Data Shape : {}" .format(joined_data.shape))

print(joined_data.iloc[-5:-1])

a=((joined_data.isnull().sum()/len(joined_data))*100).sort_values(ascending=False)

a.plot.bar()

plt.savefig('./output/figure2.png')

#lets impute BMI with its Mean as it has very less missing records

train_data['bmi']=train_data.bmi.fillna(train_data.bmi.mean())

((train_data.isnull().sum()/len(train_data))*100).sort_values(ascending=False)



'''
Handling Catagorical varriable
'''

for i in train_data.select_dtypes(exclude=['int','float']).columns:
    print('*******',i,'******') 
    print(train_data[i].value_counts())
    print('*'*30)
    print()



'''
Lets get rid off these Catagorical varriables in a way so that our machine can understand these varriables
'''
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

train_data_orignal=train_data.copy()
train_data_orignal.head()


train_data['gender']=le.fit_transform(train_data['gender'])
train_data['ever_married'] = le.fit_transform(train_data['ever_married'])
train_data['work_type']= le.fit_transform(train_data['work_type'])
train_data['Residence_type']= le.fit_transform(train_data['Residence_type'])

print(train_data.info())

train_data_without_smoke=train_data[train_data.smoking_status.isnull()]

train_data_with_smoke=train_data[train_data.smoking_status.notnull()]

train_data_without_smoke.drop('smoking_status',axis=1,inplace=True)

print("Shape of Train Data With Smoked Data: {}".format(train_data_with_smoke.shape))
print("Shape of Train Data Without Smoked Data: {}".format(train_data_without_smoke.shape))

print(train_data_without_smoke.head())




'''
Lets Encode the Smoking status fro train data
'''

train_data_with_smoke.smoking_status=le.fit_transform(train_data_with_smoke.smoking_status)
f,ax=plt.subplots(figsize=(12,10))

corr=train_data_with_smoke.corr()

sns.heatmap(train_data_with_smoke.corr(),mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(520,14,as_cmap=True),square=True, ax=ax)

plt.savefig('./output/heatmap.png')


print(train_data_orignal.head())

sns.catplot(x='smoking_status',kind='count',col='stroke',hue='gender',data=train_data_orignal,palette="Blues_d")
plt.savefig("./output/smokingStatus_VS_Count.png")

sns.catplot(x='hypertension',kind='count',col='stroke',hue='gender',data=train_data_orignal,palette="rainbow")
plt.savefig('./output/hypertensionCount.png')

sns.catplot(x='heart_disease',kind='count',col='stroke',hue='gender',data=train_data_orignal,palette="muted")
plt.savefig("./output/heartDisease_VS_Gender.png")

sns.catplot(x='ever_married',kind='count',col='stroke',hue='gender',data=train_data_orignal,palette="BuGn_r")
plt.savefig("./output/Marriage_VS_Gender.png")



#Handling missing Data

train_data_with_smoke['stroke'].value_counts()
train_data_with_smoke['stroke'].value_counts().plot.bar()
plt.savefig('./output/stroke.png')

plt.figure(figsize=(12,5))

plt.title("Distribution of age")

sns.distplot(train_data_with_smoke['age'],color='B')
plt.savefig("./output/AgeDistribution.png")




#Lets Explore the age varriable

sns.catplot(x='smoking_status',kind='count',col='stroke',data=train_data_orignal[(train_data_orignal['age']>50) & (train_data_orignal['age']<70)],palette='rainbow',hue='gender')
plt.savefig('./output/smoking_status.png')


from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN

ros=RandomOverSampler(random_state=0)
smote=SMOTE()

x_resampled,y_resampled =ros.fit_resample(train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'],train_data_with_smoke['stroke'])

print("ROS Shape Of Train Data With Smoke(Independent varriable) :   {}".format(x_resampled.shape))
print("ROS Shape Of Train Data With Smoke(dependent varriable) :   {}".format(y_resampled.shape))



x_resampled_1,y_resampled_1=ros.fit_resample(train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'],train_data_without_smoke['stroke'])

print("ROS Shape Of Train Data Without Smoke(Independent varriable) :   {}".format(x_resampled_1.shape))
print("ROS Shape Of Train Data Without Smoke(dependent varriable) :   {}".format(y_resampled_1.shape))




#Now Lets Split our Resampled Data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x_resampled,y_resampled,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(x_resampled_1,y_resampled_1,test_size=0.2)
print(X_train_1.shape)
print(X_test_1.shape)




'''
Lets Create Our Model.{Decision Tree Classifier }
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,auc,roc_auc_score,precision_score,recall_score,roc_curve

dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred=dtree.predict(X_test)

print(classification_report(y_test,pred))
print('**'*40)

print("Accuracy Score : {} ".format(accuracy_score(y_test,pred)))
print('**'*40)

print("Confusion Matrix : {} ".format(confusion_matrix(y_test,pred)))
print('**'*40)


print("Precision Score : {} ".format(precision_score(y_test,pred)))
print('**'*40)


print("Recall Score : {} ".format(recall_score(y_test,pred)))
print('**'*40)

y_pred_prob=dtree.predict_proba(X_test)[::,1]

fpr, tpr, _ = roc_curve(y_test,y_pred_prob)
auc=roc_auc_score(y_test,y_pred_prob)

plt.plot(fpr,tpr,label="AUC SCORE"+str(auc))
plt.legend(loc=4)
plt.savefig('./output/AccuracyScore.png')

#Lets Checkout The Importance Fetature 

Imp_Feature=pd.DataFrame(dtree.feature_importances_ ,index=train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print(Imp_Feature)

print(Imp_Feature.values)

Imp_Feature.plot(kind='bar')
plt.savefig('./output/barChart.png')



#Dtree Without Smoking Status111

dtree1=DecisionTreeClassifier()
dtree1.fit(X_train_1,y_train_1)
pred1=dtree1.predict(X_test_1)

print(classification_report(y_test_1,pred1))
print('**'*40)

print("Accuracy Score : {} ".format(accuracy_score(y_test_1,pred1)))
print('**'*40)

print("Confusion Matrix : {} ".format(confusion_matrix(y_test_1,pred1)))
print('**'*40)


print("Precision Score : {} ".format(precision_score(y_test_1,pred1)))
print('**'*40)


print("Recall Score : {} ".format(recall_score(y_test_1,pred1)))
print('**'*40)



y_pred_prob_1=dtree1.predict_proba(X_test_1)[::,1]

fpr, tpr, _ = roc_curve(y_test_1,y_pred_prob_1)
auc=roc_auc_score(y_test_1,y_pred_prob_1)

plt.plot(fpr,tpr,label="AUC SCORE="+str(auc))
plt.legend(loc=4)
plt.savefig('./output/AccuracyScore_2.png')


'''
Now Lets Predict on our Test Data
'''

test_data.bmi=test_data.bmi.fillna(test_data.bmi.mean())

((test_data.isnull().sum()/len(test_data))*100).sort_values(ascending=False)

test_data['gender'] = le.fit_transform(test_data['gender'])
test_data['ever_married'] = le.fit_transform(test_data['ever_married'])
test_data['work_type']= le.fit_transform(test_data['work_type'])
test_data['Residence_type']= le.fit_transform(test_data['Residence_type'])

test_data.drop(axis=1,columns=['smoking_status'],inplace=True)

test_data.info()



final_data=dtree1.predict(test_data)

Result_df=pd.DataFrame(final_data,columns=['Pred'])

print(Result_df['Pred'].value_counts())

