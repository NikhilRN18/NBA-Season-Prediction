import pandas as pd
import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Class for Team + Game Node
class TeamNode:

    def __init__(self,nodeID,gmDate,teamAbbr,teamDayOff,opptAbbr,additionalData=[]):
        self.id = nodeID
        self.date = gmDate
        self.name = teamAbbr
        self.dayOff = teamDayOff
        self.opponent = opptAbbr
        self.data = additionalData

    def getDate(self):
        return self.date
    
    def getName(self):
        return self.name
    
    def getDayOff(self):
        return self.dayOff
    
    def getOpponent(self):
        return self.opponent
   
    def getID(self):
        return self.id
    
    def setData(self,additionalData):
        self.data = additionalData



# Default Functions
def getSeason(gmDate,seasonOpeners):
    gmDateInt = int("".join(gmDate.split("-")))
    currYear = 0
    # print(gmDateInt)
    while(currYear <= 5 and gmDateInt >= seasonOpeners[currYear]):
        currYear += 1
    return str(currYear + 2011)

def getPrevGame(node,teamSumm):
    return teamSumm[node.getName()][-1]

def getOpponentNode(node, teamSumm):
    if node.getOpponent() in list(teamSumm.keys()):
        opponentGames = teamSumm[node.getOpponent()]
        for i in opponentGames:
            if i.getDate() == node.getDate():
                return i
    return None



# Homogenous Graph Construction
def makeGraphs(data,seasonOpeners, toKeep):
    seasonDict = {"2012":[],"2013":[],"2014":[],"2015":[],"2016":[],"2017":[]}
    graphs = []
    outcomes = {"2012":[],"2013":[],"2014":[],"2015":[],"2016":[],"2017":[]}
    node_mapping = {}
    # Split data by seasons
    for _,i in data.iterrows():
        currNode = TeamNode(_,i['gmDate'],i['teamAbbr'],i['teamDayOff'],i['opptAbbr'])
        currNode.setData(i[toKeep])
        seasonDict[getSeason(i['gmDate'],seasonOpeners)].append(currNode)
        outcomes[getSeason(i['gmDate'],seasonOpeners)].append(int(i['teamRslt'] == 'Win'))
        node_mapping[currNode.getID()] = currNode
    # Create Edges
    for _, year_data in seasonDict.items():
        yearByTeam = {}
        u = []
        v = []
        nm = {}
        node_id = 0
        # Map nodes to new IDs
        for i in year_data:
            nm[i.getID()] = node_id
            node_id += 1
        for i in year_data:
            curr_node_id = nm[i.getID()]
            # Add Previous Game
            if i.getName() not in yearByTeam.keys():
                yearByTeam[i.getName()] = [i]
            else:
                prev_node = getPrevGame(i, yearByTeam)
                if prev_node:
                    u.append(curr_node_id)
                    v.append(nm[prev_node.getID()])
                yearByTeam[i.getName()].append(i)   
            # Add Opponent
            opp = getOpponentNode(i, yearByTeam)
            if opp is not None:
                opp_node_id = nm[opp.getID()]
                u.append(curr_node_id)
                v.append(opp_node_id)
        u_tensor = torch.tensor(u, dtype=torch.int64)
        v_tensor = torch.tensor(v, dtype=torch.int64)
        currGraph = dgl.graph((u_tensor, v_tensor))
        graphs.append(dgl.to_bidirected(currGraph))
    return graphs, list(outcomes.values()), node_mapping



# Variable Definitions
boxScores_raw = pd.read_csv("/Users/rishabhchhabra/Library/CloudStorage/GoogleDrive-1573108@fcpsschools.net/My Drive/12th Grade/ML/Semester 2 Project/2012-18_teamBoxScore.csv")
boxScores = boxScores_raw[["gmDate","teamAbbr","teamDayOff","teamLoc","teamAST","teamTO","teamSTL","teamBLK","teamPF","teamFGA","teamFGM","teamFG%",
                           "team2PA","team2PM","team2P%","team3PA","team3PM","team3P%","teamFTA","teamFTM","teamFT%","teamORB","teamDRB","teamTRB","teamTREB%",
                           "teamASST%","teamTS%","teamEFG%","teamOREB%","teamDREB%","teamTO%","teamSTL%","teamBLK%","teamBLKR","teamPPS","teamFIC","teamFIC40",
                           "teamOrtg","teamDrtg","teamEDiff","teamPlay%","teamAR","teamAST/TO","teamSTL/TO","poss","pace","opptAbbr","teamRslt"]]
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
    bs1 = bs.drop(['gmDate','teamAbbr','opptAbbr','teamRslt'],axis=1).reset_index()
    bs1.drop(bs1.columns[[0]], axis=1, inplace=True)
    bs1['teamLoc'] = (bs1['teamLoc'] == 'Home').astype(int)
    y1['teamRslt'] = (y1['teamRslt'] == 'Win').astype(int)
    allBoxScores.append(bs1)
    allResults.append(y1)

# Random Forest Feature Selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(allBoxScores[1], allResults[1])
importances = rf.feature_importances_
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(allBoxScores[1].columns, importances)]
kept_features = [feature for feature, importance in feature_importances if importance > 0.1]
print("Features kept:",kept_features)
tempBoxScores = []
for i in allBoxScores:
    tempBoxScores.append(i[kept_features])
allBoxScores.clear()
allBoxScores = tempBoxScores


accuracies = []
for n in range(len(allBoxScores)):
    currBoxScores = allBoxScores[n]
    currResults = allResults[n]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for i, (train_index, test_index) in enumerate(sss.split(currBoxScores, currResults)):
        X_train = currBoxScores.loc[train_index]
        y_train = pd.DataFrame(currResults.loc[train_index])
        X_test = currBoxScores.loc[test_index]
        y_test = pd.DataFrame(currResults.loc[test_index])

    seasonAccs = []
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_predictions)
    # print("Decision Tree Accuracy:", dt_acc)
    seasonAccs.append(dt_acc)

    # SVM
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train['teamRslt'])
    svm_predictions = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test['teamRslt'], svm_predictions)
    # print("SVM Accuracy:", svm_acc)
    seasonAccs.append(svm_acc)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = (lr_model.predict(X_test) > 0.5).astype(int)
    lr_acc = accuracy_score(y_test, lr_predictions)
    # print("Linear Regression Classification Accuracy:", lr_acc)
    seasonAccs.append(lr_acc)
    accuracies.append(seasonAccs)



# GCN MODEL

seasonGraphs,outcomes,node_mapping = makeGraphs(boxScores,seasonStartDates,kept_features)
class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

def create_features_labels(graph, season_outcomes):
    features_list = []
    labels_list = []
    for node_id in range(graph.number_of_nodes()):
        node = node_mapping[node_id]
        additional_data = []
        for x in node.data:
            try:
                additional_data.append(float(x))
            except ValueError:
                additional_data.append(float(int(x == 'Home')))
        team_day_off = float(node.getDayOff())
        features = np.array(additional_data + [team_day_off])
        features_list.append(features)
        labels_list.append(season_outcomes[node_id])
    
    features = torch.tensor(features_list, dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels

def split_graph(graph):
    total_nodes = graph.number_of_nodes()
    indices = np.random.permutation(total_nodes)
    train_size = int(0.7 * total_nodes)
    val_size = int(0.1 * total_nodes)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask

# Parameters
num_features = len(node_mapping[0].data) + 1
hidden_size = 64
num_classes = 2
learning_rate = 0.01
num_epochs = 500

gcnAccs = []

for i, season_graph in enumerate(seasonGraphs):
    model = GCNModel(num_features, hidden_size, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    season_outcomes = outcomes[i]
    features, labels = create_features_labels(season_graph, season_outcomes)
    train_mask, val_mask, test_mask = split_graph(season_graph)

    # Training
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(season_graph, features)
        loss = criterion(outputs[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

    # Testing
    model.eval()
    with torch.no_grad():
        outputs = model(season_graph, features)
        _, predicted = torch.max(outputs[test_mask], 1)
        accuracy = accuracy_score(labels[test_mask].cpu(), predicted.cpu())
        # print("GCN Classification Accuracy:", accuracy)
        gcnAccs.append(accuracy)


for i in range(len(gcnAccs)): accuracies[i].append(gcnAccs[i])

# Display
print("Accuracies:")
finalAccs = pd.DataFrame(np.array(accuracies),columns=['DT','SVM','LR','GCN'])
finalAccs.insert(0,'Year',['2012-13','2013-14','2014-15','2015-16','2016-17','2017-18'])
avg_accs = {'Year':'Average', 'DT': finalAccs.loc[:, 'DT'].mean(), 'SVM': finalAccs.loc[:, 'SVM'].mean(), 'LR': finalAccs.loc[:, 'LR'].mean(), 'GCN': finalAccs.loc[:, 'GCN'].mean()}
finalAccs = finalAccs.append(avg_accs, ignore_index=True)
print(finalAccs)


