# Importing the libraries
import pandas as pd
import pickle
import numpy as np
#from sklearn.model_selection import train_test_split
np.warnings.filterwarnings('ignore')

#dataset = pd.read_csv('hue1.csv')
#dataset = dataset.replace(to_replace="N", value=0)
#dataset = dataset.replace(to_replace="D", value=1)
#dataset = dataset.replace(to_replace="M", value=2)

episodes = pd.read_csv('episodios.csv', sep=';')
episodes = episodes.dropna(axis=0)
episodes.columns = ['patient', 'start', 'end', 'episode']
episodes = episodes.replace(to_replace="DEPRESIÓN", value=1)
episodes = episodes.replace(to_replace="MANIA", value=2)
from datetime import datetime

for index, row in episodes.iterrows():
    episodes['start'][index] = datetime.strptime(row.start, '%d/%m/%Y')
    episodes['end'][index] = datetime.strptime(row.end, '%d/%m/%Y')

interviews = pd.read_csv('diario.csv', sep=';')
interviews.columns = ['mood', 'motivation', 'attention', 'irritability', 'anxiety',
                      'sleep_quality', 'menstrual_cycle', 'nr_cigarettes', 'caffeine',
                      'alcohol', 'other_drugs', 'wake up time', 'going to bed time',
                      'patient', 'date']
interviews = interviews.replace(to_replace="NO", value="No")
interviews = interviews.replace(to_replace="SI", value="Si")
interviews = interviews.replace(to_replace="No", value=0)
interviews = interviews.replace(to_replace="Si", value=1)
interviews = interviews.replace(to_replace=":", value="")
interviews['wake up time'] = interviews['wake up time'].str.replace(':','')
interviews['going to bed time'] = interviews['going to bed time'].str.replace(':','')
interviews = interviews.apply(pd.to_numeric, errors='ignore')
interviews.loc[interviews['going to bed time'] < interviews['wake up time'], 'going to bed time'] = interviews['going to bed time'] + 2400
interviews['active_time'] = abs((interviews['wake up time'] - interviews['going to bed time']).astype(int))
interviews = interviews.drop('wake up time', axis=1)
interviews = interviews.drop('going to bed time', axis=1)
interviews = interviews.drop('menstrual_cycle', axis=1)

interviews = interviews.dropna(axis=0)
from datetime import datetime

for index, row in interviews.iterrows():
    interviews['date'][index] = datetime.strptime(row.date, '%d/%m/%Y')

interviews = interviews.sort_values('date')

def checkEpisode(date, patient):
    episode = 'N'
    ep = episodes.loc[episodes['patient'] == patient]
    for index, row in ep.iterrows():
        if date >= row.start and date < row.end:
            episode = row.episode
    return episode

interviews_episodes = interviews.copy()
for index, row in interviews_episodes.iterrows():
    interviews_episodes.at[index, 'episode'] = checkEpisode(row.date, row.patient)

interviews_episodes = interviews_episodes.drop('patient', axis= 1)
interviews_episodes = interviews_episodes.drop('date', axis=1)
interviews_episodes = interviews_episodes.replace(to_replace="N", value=0)


X = interviews_episodes.iloc[:, :11]
print(X)
y = interviews_episodes.iloc[:, -1]
print(y)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
import catboost as cat
catm = cat.CatBoostClassifier(one_hot_max_size=30,
                              max_depth=7,
                              learning_rate=0.062030,
                              iterations=500,

                              loss_function='MultiClass',
                              )

#Fitting model with trainig data
catm.fit(X, y)

# Saving model to disk
pickle.dump(catm, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[-2,-3,2,1,2,3,26,120,0,0,1695]]))