import csv
import math
import operator
def loadDataset(filename,split,trainingSet=[],testSet=[]):
        with open(filename,'rt') as csvfile:
                lines = csv.reader(csvfile)
                dataset = list(lines)
                for x in range(len(dataset)-1):
                        for y in range(32):
                                dataset[x][y]=float(dataset[x][y])
                        if x<split*len(dataset):
                            trainingSet.append(dataset[x])
                        else:
                            testSet.append(dataset[x])
def calcDistance(i1,i2,length):
        distance=0
        for x in range(length):
                distance+=pow((i1[x]-i2[x]),2)
        return math.sqrt(distance)

def getNeighbors(trainingSet,testInstance,k):
        distances=[]
        length=len(testInstance)-1
        for x in range(len(trainingSet)):
                dist=calcDistance(testInstance,trainingSet[x],length)
                distances.append((trainingSet[x],dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors=[]
        for x in range(k):
                neighbors.append(distances[x][0])
        return neighbors
def getResponse(neighbors):
        classVotes={}  # dictionary
        for x in range(len(neighbors)):
                response=neighbors[x][-1]
                if response in classVotes:
                        classVotes[response]+=1
                else:
                        classVotes[response]=1
        sortedVotes=sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
        return sortedVotes[0][0]

def getAccuracy(testSet,predictions):
        correct=0
        for x in range(len(testSet)):
                if testSet[x][-1]==predictions[x]:
                        correct+=1
        return (correct/float(len(testSet)))*100.0
def main():
        trainingSet=[]
        testSet=[]
        predictions=[]
        print('Loading Dataset')
        # load the dataset
        loadDataset('newfile.txt',0.67,trainingSet,testSet)        
        k=3   # choose k value
        i=1;
        for x in range(len(testSet)):
                neighbors=getNeighbors(trainingSet,testSet[x],k)
                result=getResponse(neighbors)
                predictions.append(result)
                # printing k lie
                print(repr(i)+'  predicted='+repr(result)+', actual='+repr(testSet[x][-1]))
                i=i+1
        accuracy=getAccuracy(testSet,predictions)
        print('Accuracy: '+repr(accuracy)+'%')
main()
