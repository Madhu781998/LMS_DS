
import pandas as pd
import numpy as np

data1 = pd.read_csv("C:/Users/ketan/OneDrive/Desktop/360digiTMG/14 Recommmendation Engine/Hands-on Material/game.csv")
data1.head()

#Creating a sparse matrix 
matrix = data1.pivot_table( columns = 'game', index = 'userId', values = 'rating')
#it will have many nan value

#remove some of the gamers who don’t play that many games
matrix = matrix.dropna(thresh = 2, axis = 0) #code snippet basically counts those who have less than 2 games played and drops their rows
matrix

#These are the most high rated or popular games
data1.groupby('game')['rating'].mean().sort_values(ascending = False).head(10)

#fill the null value with zeros
matrix = data1.pivot_table(columns = 'game', index = 'userId', values = 'rating', fill_value = 0)

#standardize the row values
def std(i):
    new_row = (i - i.min()) / (i.max() - i.min())
    return new_row

matrix_std = matrix.apply(std)

def gameRec(g):
    data = matrix_std[g] #The data variable is what I used to return our values from our centered matrix for a particular game
    return data
#Calculate correlation 
data = matrix.corrwith(data1).dropna()

#create a DF to show how many times each game has been played and the mean rating has been given
gameData = data1.groupby('game').agg({'rating': [np.size, np.mean]})

#Filter out any game played by less than 100 players.
gameSim = gameData['rating']['size'] >= 5

df = gameData[gameSim].join(pd.DataFrame(data, columns=['similarity']))
df.sort_values(['similarity'], ascending=False)[:7]
gameRec('Marvel Pinball')

gameRec("Tony Hawk's Pro Skater 2")
