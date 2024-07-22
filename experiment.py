import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def getSeason(gmDate,seasonOpeners):
    gmDateInt = int("".join(gmDate.split("-")))
    currYear = 0
    # print(gmDateInt)
    while(currYear <= 5 and gmDateInt >= seasonOpeners[currYear]):
        currYear += 1
    return str(currYear + 2011)


# Variable Definitions
boxScores_raw = pd.read_csv("/Users/rishabhchhabra/Library/CloudStorage/GoogleDrive-1573108@fcpsschools.net/My Drive/12th Grade/ML/Semester 2 Project/2012-18_teamBoxScore.csv")
boxScores = boxScores_raw[["gmDate","teamAbbr","opptAbbr","teamRslt"]]
seasonStartDates = [20121030,20131029,20141028,20151027,20161025,20171017]
boxScores2012 = boxScores.loc[boxScores['gmDate'] < '2013-10-29']
boxScores2013 = boxScores.loc[(boxScores['gmDate'] >= '2013-10-29') & (boxScores['gmDate'] < '2014-10-28')]
boxScores2014 = boxScores.loc[(boxScores['gmDate'] >= '2014-10-28') & (boxScores['gmDate'] < '2015-10-27')]
boxScores2015 = boxScores.loc[(boxScores['gmDate'] >= '2015-10-27') & (boxScores['gmDate'] < '2016-10-25')]
boxScores2016 = boxScores.loc[(boxScores['gmDate'] >= '2016-10-25') & (boxScores['gmDate'] < '2017-10-17')]
boxScores2017 = boxScores.loc[boxScores['gmDate'] >= '2017-10-17']
allBoxScores = []
allResults = []
for bs in [boxScores2012,boxScores2013,boxScores2014,boxScores2015,boxScores2016,boxScores2017]:
    y1 = pd.DataFrame(bs['teamRslt']).reset_index().drop('index',axis=1)
    bs1 = bs.drop(['gmDate','teamRslt'],axis=1).reset_index()
    bs1.drop(bs1.columns[[0]], axis=1, inplace=True)
    y1['teamRslt'] = (y1['teamRslt'] == 'Win').astype(int)
    allBoxScores.append(bs1)
    allResults.append(y1)

standings2012 = ['MIA','OKC','SA','DEN','LAC','MEM','NY','IND','BKN','GS','CHI','LAL','HOU','ATL','UTA','BOS','DAL','MIL','PHI','TOR','POR','MIN','DET','WAS','SAC','NO','PHO','CLE','CHA','ORL']
standings2013 = ['SA','OKC','LAC','IND','MIA','HOU','POR','GS','MEM','DAL','TOR','CHI','PHO','WAS','BKN','CHA','MIN','ATL','NY','DEN','NO','CLE','DET','SAC','LAL','BOS','UTA','ORL','PHI','MIL']
standings2014 = ['GS','ATL','HOU','LAC','MEM','SA','CLE','POR','CHI','DAL','TOR','WAS','NO','OKC','MIL','BOS','PHO','BKN','IND','UTA','MIA','CHA','DET','DEN','SAC','ORL','LAL','PHI','NY','MIN']
standings2015 = ["GS",  # Golden State Warriors
    "SA",  # San Antonio Spurs
    "CLE",  # Cleveland Cavaliers
    "TOR",  # Toronto Raptors
    "OKC",  # Oklahoma City Thunder
    "LAC",  # Los Angeles Clippers
    "ATL",  # Atlanta Hawks
    "BOS",  # Boston Celtics
    "CHA",  # Charlotte Hornets
    "MIA",  # Miami Heat
    "IND",  # Indiana Pacers
    "DET",  # Detroit Pistons
    "POR",  # Portland Trail Blazers
    "DAL",  # Dallas Mavericks
    "MEM",  # Memphis Grizzlies
    "CHI",  # Chicago Bulls
    "HOU",  # Houston Rockets
    "WAS",  # Washington Wizards
    "UTA",  # Utah Jazz
    "ORL",  # Orlando Magic
    "DEN",  # Denver Nuggets
    "MIL",  # Milwaukee Bucks
    "SAC",  # Sacramento Kings
    "NY",  # New York Knicks
    "NO",  # New Orleans Pelicans
    "MIN",  # Minnesota Timberwolves
    "PHO",  # Phoenix Suns
    "BKN",  # Brooklyn Nets
    "LAL",  # Los Angeles Lakers
    "PHI"   # Philadelphia 76ers
    ]
standings2016 = [
    "GS",  # Golden State Warriors
    "SA",  # San Antonio Spurs
    "HOU",  # Houston Rockets
    "BOS",  # Boston Celtics
    "CLE",  # Cleveland Cavaliers
    "LAC",  # Los Angeles Clippers
    "TOR",  # Toronto Raptors
    "UTA",  # Utah Jazz
    "WAS",  # Washington Wizards
    "OKC",  # Oklahoma City Thunder
    "ATL",  # Atlanta Hawks
    "MEM",  # Memphis Grizzlies
    "IND",  # Indiana Pacers
    "MIL",  # Milwaukee Bucks
    "CHI",  # Chicago Bulls
    "POR",  # Portland Trail Blazers
    "MIA",  # Miami Heat
    "DEN",  # Denver Nuggets
    "DET",  # Detroit Pistons
    "CHA",  # Charlotte Hornets
    "NO",  # New Orleans Pelicans
    "DAL",  # Dallas Mavericks
    "SAC",  # Sacramento Kings
    "MIN",  # Minnesota Timberwolves
    "NY",  # New York Knicks
    "ORL",  # Orlando Magic
    "PHI",  # Philadelphia 76ers
    "LAL",  # Los Angeles Lakers
    "PHO",  # Phoenix Suns
    "BKN"   # Brooklyn Nets
]
standings2017 = [
    "HOU",  # Houston Rockets
    "TOR",  # Toronto Raptors
    "GS",  # Golden State Warriors
    "BOS",  # Boston Celtics
    "PHI",  # Philadelphia 76ers
    "CLE",  # Cleveland Cavaliers
    "POR",  # Portland Trail Blazers
    "IND",  # Indiana Pacers
    "NO",  # New Orleans Pelicans
    "OKC",  # Oklahoma City Thunder
    "UTA",  # Utah Jazz
    "MIN",  # Minnesota Timberwolves
    "SA",  # San Antonio Spurs
    "DEN",  # Denver Nuggets
    "MIA",  # Miami Heat
    "MIL",  # Milwaukee Bucks
    "WAS",  # Washington Wizards
    "LAC",  # Los Angeles Clippers
    "DET",  # Detroit Pistons
    "CHA",  # Charlotte Hornets
    "LAL",  # Los Angeles Lakers
    "NY",  # New York Knicks
    "BKN",  # Brooklyn Nets
    "CHI",  # Chicago Bulls
    "SAC",  # Sacramento Kings
    "ORL",  # Orlando Magic
    "ATL",  # Atlanta Hawks
    "DAL",  # Dallas Mavericks
    "MEM",  # Memphis Grizzlies
    "PHO"   # Phoenix Suns
]
allStandings = [standings2012,standings2013,standings2014,standings2015,standings2016,standings2017]

accuracies = []
for idx,val in enumerate(allBoxScores):
    bs = val
    res = allResults[idx]['teamRslt']
    sta = allStandings[idx]
    pred = []
    for _,i in bs.iterrows():
        pred.append(int(sta.index(i['teamAbbr']) < sta.index(i['opptAbbr'])))
    accuracies.append(accuracy_score(res,pred))

print(accuracies)
print("Accuracies:")
finalAccs = pd.DataFrame(np.array(accuracies),columns=['By Ranking'])
finalAccs.insert(0,'Year',['2012-13','2013-14','2014-15','2015-16','2016-17','2017-18'])
avg_accs = {'Year':'Average', 'By Ranking': finalAccs.loc[:, 'By Ranking'].mean()}
finalAccs = finalAccs.append(avg_accs, ignore_index=True)
print(finalAccs)

