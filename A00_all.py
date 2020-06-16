import os
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics

class DataProccessor:
    def __init__(self, AKIDataDir):
        self.__demoSubFeatureNum = 4
        self.__vitalSubFeatureNum = 8
        self.__labSubFeatureNum = 817
        self.__ccsSubFeatureNum = 2621
        self.__pxSubFeatureNum = 15606
        self.__medSubFeatureNum = 11539
        self.__incompAKIFeatureName = None
        self.__AKIFeatureName = None
        
        self.setAKIDaDir(AKIDataDir)
    
    def setAKIDaDir(self, AKIDataDir):
        self.__AKIDataDir = AKIDataDir
        self.__rawAKIData = None
        self.__AKIData = None
        self.__filteredAKIData = None
    
    def getAKIDaDir(self):
        return self.__AKIDataDir
    
    def getRawAKIData(self):
        if self.__rawAKIData == None:
            self.__initRawAKIData()
        return self.__rawAKIData
    
    def getAKIFeatureName(self):
        if self.__AKIFeatureName == None:
            self.__initAKIFeatureName()
        return self.__AKIFeatureName
    
    def getIncompFeatureName(self):
        if self.__incompAKIFeatureName == None:
            self.__initIncompAKIFeature()
        return self.__incompAKIFeatureName
    
    def preprocess(AKIData): 
        appendData = pd.DataFrame()
        markedCols = []
        tmpList = AKIData.columns[: 4+8]
        for column in tmpList:
            if AKIData[column].dtype == "object":
                newCols = pd.get_dummies(AKIData[column], prefix=column, prefix_sep='_')
                appendData = pd.concat([appendData, newCols], axis=1)
                markedCols.append(column)
        AKIData.drop(markedCols, axis=1, inplace=True)
        AKIData = pd.concat([appendData, AKIData], axis=1)
        return AKIData

    def getAKIData(self, deleteEmptyCol=False):
        markedIndex = []
        if self.__AKIData is None:
            self.__initRawAKIData()
            AKIDataArray = [self.__getPreprocessedDataFrom(aRawData) for aRawData in self.__rawAKIData.values]
            AKIFeatureName = self.getAKIFeatureName()
            self.__AKIData = pd.DataFrame(AKIDataArray)
            self.__AKIData.columns = AKIFeatureName
        if deleteEmptyCol:
            # 出除无用列（-2是为了保留最后一列：label）
            for column in self.__AKIData.columns[: -2]:
                if (self.__AKIData[column].nunique() == 1):
                    markedIndex.append(column)
        return self.__AKIData.drop(markedIndex, axis=1, inplace=False)
    
    def getFilteredData(self, deleteEmptyCol=False):
        markedIndex = []
        if self.__filteredAKIData is None:
            inconpFeatureName = self.getIncompFeatureName()
            self.__filteredAKIData = self.getAKIData()[inconpFeatureName]
        if deleteEmptyCol:
            for column in self.__filteredAKIData.columns[: -2]:
                if (self.__filteredAKIData[column].nunique() == 1):
                    markedIndex.append(column)
        return self.__filteredAKIData.drop(markedIndex, axis=1, inplace=False)

    def __initRawAKIData(self):
        rawAKIdata = []
        for fileName in os.listdir(self.__AKIDataDir):
            if os.path.isfile(os.path.join(self.__AKIDataDir, fileName)):
                rawAKIdata.extend(pd.read_pickle(os.path.join(self.__AKIDataDir, fileName)))
        self.__rawAKIData = pd.DataFrame(rawAKIdata)
        self.__rawAKIData.columns = ['demo', 'vital', 'lab', 'comorbidity', 'procedure', 'med', 'AKI_label']

    def __initAKIFeatureName(self):
        self.__AKIFeatureName = []
        self.__AKIFeatureName.extend(["DEMO_age", "DEMO_hispanic", "DEMO_race", "DEMO_sex"])
        self.__AKIFeatureName.extend(["VITAL_height", "VITAL_weight", "VITAL_BMI", 
                            "VITAL_smoking", "VITAL_tabacco", "VITAL_tabaccoType", "VITAL_SBP", "VITAL_DBP"])
        self.__AKIFeatureName.extend(["Lab_" + str(i) for i in range(1, self.__labSubFeatureNum+1)])
        self.__AKIFeatureName.extend(["CCS_" + str(i) for i in range(1, self.__ccsSubFeatureNum+1)])
        self.__AKIFeatureName.extend(["PX_" + str(i) for i in range(1, self.__pxSubFeatureNum+1)])
        self.__AKIFeatureName.extend(["Med_" + str(i) for i in range(1, self.__medSubFeatureNum+1)])
        self.__AKIFeatureName.append("Label")
    
    def __initIncompAKIFeature(self):
        self.__incompAKIFeatureName = []
        self.__incompAKIFeatureName.extend(["DEMO_age", "DEMO_hispanic", "DEMO_race", "DEMO_sex"])
        self.__incompAKIFeatureName.extend(["VITAL_height", "VITAL_weight", "VITAL_BMI", "VITAL_smoking", 
                               "VITAL_tabacco", "VITAL_tabaccoType"])
        self.__incompAKIFeatureName.extend(["CCS_" + str(i) for i in range(1, self.__ccsSubFeatureNum+1)])
        self.__incompAKIFeatureName.append("Label")

    def __getPreprocessedDataFrom(self, aPatientRawData):
        # 获取发病日期。([6]表示提取第7个大类特征，[0]表示第1条记录)
        label, firstAKITime = aPatientRawData.T[6][0] 
        firstAKITime = float(firstAKITime)
        label = 1 if label!="0" else 0
        
        [demo, vital, lab, comorbidity, procedure, med, AKI_label] = [item for item in aPatientRawData.T]

        # 抽取人口统计学数据（1维）：Age_Hispanic_Race_Sex。Age解析为数值，剩余的是离散变量
        demoData = []
        demoData.append(float(demo[0]))
        demoData.extend(demo[1:])

        # 抽取体征数据(3维)：Height_Weight_BMI_Smoking_Tabacco_TabaccoType_SBP_DBP
        # Height_Weight_BMI_SBP_DBP解析为数值，Smoking_Tabacco_TabaccoType为离散变量
        vitalData = []
        for index, subFeatureData in enumerate(vital):
            minTimeInterval, targetValue = float('inf'), float(subFeatureData[0][0])
            for recode in subFeatureData:
                if len(recode) < 2:
                     recode = ['0', 0]       # 当没有时间记录，说明患者在这一项上为空。直接赋0
                value, time = recode 
                time = float(time)
                timeInterval = firstAKITime - time
                if timeInterval>0 and timeInterval<minTimeInterval:
                    minTimeInterval = timeInterval
                    if index==3 or index==4 or index==5: # 将Smoking_Tabacco_TabaccoType保留为离散变量
                        targetValue = str(value)
                    else:
                        targetValue = float(value)
            vitalData.append(targetValue)

        # 抽取实验室数据（4维）：Lab1_Lab2_Lab3_...
        labData = [0 for i in range(0, self.__labSubFeatureNum)] # 初始化lab数据为全0
        for subFeatureData in lab:
            isEmptyData = (subFeatureData == [[['0']]])
            if isEmptyData:
                break

            subFeatureName, subFeatureRecodeList = subFeatureData[0][0][0], subFeatureData[1]
            subFeatureIndex = int(subFeatureName.split("lab")[1])

            minTimeInterval, value = float('inf'), float(subFeatureRecodeList[0][0])
            for recode in subFeatureRecodeList:
                valueStr, _, timeStr = recode
                time = float(timeStr)
                timeInterval = firstAKITime - time

                if timeInterval>0 and timeInterval<minTimeInterval:
                    minTimeInterval, targetValue = timeInterval, float(valueStr)
            labData[subFeatureIndex-1] = targetValue  # 在特征表中lab是从1开始的，故减去1

        # 抽取共病（Comorbidity）数据（3）：CCS1_CCS2_CCS3_...
        ccsData = [0 for i in range(0, self.__ccsSubFeatureNum)]
        for subFeatureData in comorbidity:
            isEmptyData = (subFeatureData == [['0']])
            if isEmptyData:
                break
            subFeatureName = subFeatureData[0][0]
            subFeatureIndex = int(subFeatureName.split("ccs")[1])
            ccsData[subFeatureIndex-1] = 1  # 这里采用第一种处理方式：只要有记录，则标记为发过病

        # 抽取过程（procedure）数据（3维）：PX1_PX2_PX3_
        pxData = [0 for i in range(0, self.__pxSubFeatureNum)]
        for subFeatureData in procedure:
            isEmptyData = (subFeatureData == [['0']])
            if isEmptyData:
                break
            subFeatureName = subFeatureData[0][0]
            subFeatureIndex = int(subFeatureName.split("px")[1])
            pxData[subFeatureIndex-1] = 1  # 这里采用第一种处理方式：只要有记录，则标记为发过病

        # 抽取药物数据（4）：Med1_Med2_Med3_...
        medData = [0 for i in range(0, self.__medSubFeatureNum)] # 初始化med数据为全0
        for subFeatureData in med:
            isEmptyData = (subFeatureData == [[['0']]])
            if isEmptyData:
                break
            subFeatureName, subFeatureRecodeList = subFeatureData[0][0][0], subFeatureData[1]
            subFeatureIndex = int(subFeatureName.split("med")[1])

            minTimeInterval, value = float('inf'), float(subFeatureRecodeList[0][0])
            for recode in subFeatureRecodeList:
                valueStr, timeStr = recode
                time = float(timeStr)
                timeInterval = firstAKITime - time
                if timeInterval>0 and timeInterval<minTimeInterval:
                    minTimeInterval, targetValue = timeInterval, float(valueStr)
            medData[subFeatureIndex-1] = targetValue  # 在特征表中lab是从1开始的，故减去1
        return demoData + vitalData + labData + ccsData + pxData + medData + [label] # TODO 这里会把全部元素作为字符串（因为元素中存在字符串）

print("=========================================== begin work =======================================================")
home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
# home = "/home/xinhos"
arrayDataDir = os.path.join(home, "data/AKIData/dataArray")
dataDir = os.path.join(home, arrayDataDir, "data.pkl") 
dataFilteredDir = os.path.join(home, arrayDataDir, "dataFiltered.pkl")
if os.path.exists(dataDir):
    print("Read the data saved under ", arrayDataDir)
    data = pd.read_pickle(dataDir)
    dataFiltered = pd.read_pickle(dataFilteredDir)
else :
    AKIDataDir = os.path.join(home, "data/AKIData/listData/")
    dataProccessor = DataProccessor(AKIDataDir)
    data = dataProccessor.getAKIData(deleteEmptyCol=True)
    dataFiltered = dataProccessor.getFilteredData(deleteEmptyCol=True)

    data = DataProccessor.preprocess(data)
    dataFiltered = DataProccessor.preprocess(dataFiltered)

    print("=============================================================================================================")
    print("Array data saving")
    data.to_pickle(os.path.join(arrayDataDir, "data.pkl"))
    dataFiltered.to_pickle(os.path.join(arrayDataDir, "dataFiltered.pkl"))
    print("Array data saved")

featureName = dataFiltered.columns[:-1]  # 去除label
data, dataFiltered = data.values, dataFiltered.values

# 删除数据中带有nan、inf的行
print("============= data ==============")
print("begin", data.shape)
infIndex = np.isinf(data).any(axis=1)
nanIndex = np.isnan(data).any(axis=1)
data = data[~infIndex, :]
data = data[~nanIndex, :]
print("end", data.shape)
print("============= dataFiltered ================")
print("begin", dataFiltered.shape)
infIndex = np.isinf(dataFiltered).any(axis=1)
nanIndex = np.isnan(dataFiltered).any(axis=1)
dataFiltered = dataFiltered[~infIndex, :]
dataFiltered = dataFiltered[~nanIndex, :]
print("end", dataFiltered.shape)


# 1. 获得基线AUC
print("=============================================== get base auc ==================================================")
aucList = []
kfold = KFold(n_splits=10)
for trainDataIndex, testDataIndex in kfold.split(data):
    trainData, testData = data[trainDataIndex, :], data[testDataIndex, :]
    trainX, trainY = trainData[:, :-1], trainData[:, -1]
    testX, testY = testData[:, :-1], testData[:, -1]
    GBDTClf = GradientBoostingClassifier().fit(trainX, trainY)
    predictedTestY = GBDTClf.predict(testX)
    auc = roc_auc_score(testY, predictedTestY)
    aucList.append(auc)
baseAUC = np.average(aucList)
print("aucList:", aucList)
print("baseAUC：", baseAUC)

# 2. 使用CART进行亚组分组，并获取亚组
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
subgroups = {}
trainX, trainY = dataFiltered[:, :-1], dataFiltered[:, -1]
print("============================================= get subgroup ==================================================")
param = {
    "criterion": "gini",
    "min_samples_leaf": 200,
    "min_samples_split": 500
}
cartClf = DecisionTreeClassifier(**param).fit(trainX, trainY)

# 保存决策树结构
dotFilePath = os.path.join(home, "data/AKIData/dataArray", "tree.dot")
with open(dotFilePath, "w", encoding="utf-8") as file:
    file = export_graphviz(cartClf, out_file = file, feature_names = featureName, 
                           filled = True, rounded = True, special_characters = True)
print("Save location of decision tree structure：", dotFilePath)                           
# 获得亚组
subgroupIndex = cartClf.apply(dataFiltered[:, :-1])
for sampleIndex, groupIndex in enumerate(subgroupIndex):
    if subgroups.get(groupIndex) is None:
        subgroups[groupIndex] = []
    subgroups[groupIndex].append(sampleIndex)

print("Number of subgroups：", len(subgroups))
tmpSavePath = os.path.join(home, "data/AKIData/dataArray", "subgroups.pkl")
with open(tmpSavePath, "wb") as file:
    pickle.dump(subgroups, file)
print("Save path of subgroup data:", tmpSavePath)

# 3. 在得到的每个亚组上使用GBDT建模，并计算总体AUC，以及输出AUC变化情况
print("================================================ get subgroup auc ==============================================")
overallAUC = 0.0
subgroupAUCs = {}
preList = []
testList = []
kfold3 = KFold(n_splits=3)
for subgroupName, subgroup in subgroups.items():
    print("build model for subgroup", "%.3f"%subgroupName, "(", len(subgroup),")", end="")
    subgroup = np.array(subgroup)
    subgroupData = data[subgroup]
    aucList = []
    for trainDataIndex, testDataIndex in kfold3.split(subgroupData):
        trainData, testData = subgroupData[trainDataIndex, :], subgroupData[testDataIndex, :]
        trainX, trainY = trainData[:, :-1], trainData[:, -1]
        testX, testY = testData[:, :-1], testData[:, -1]
        GBDTClf = GradientBoostingClassifier().fit(trainX, trainY)
        predictedTestY = GBDTClf.predict(testX)
#         fpr, tpr, thresholds = metrics.roc_curve(testY, predictedTestY, pos_label=2)
#         auc = metrics.auc(fpr, tpr)
        # 在小样本情况下，有可能预测的是完全一样的，会出现只有一个类的情况，此时调用roc_auc_score会出错
        auc = metrics.roc_auc_score(testY, predictedTestY)         
        aucList.append(auc)
    subgroupAUCs[subgroupName] = aucList
    print("average AUC:", np.average(aucList), ",AUC value list：", ["%.4f"%auc for auc in aucList])
 
# 4. 输出亚组信息&保存数据
print("================================================= print info =================================================")
count = 0
weights = []
size = len(dataFiltered)
for subgroupName, subgroupAUC in subgroupAUCs.items():
    subgroupSize = len(subgroups[subgroupName])
    weight = float(subgroupSize)/float(size)
    print("size:", subgroupSize, "weight:", "%.4f"%weight, "auc:", "%.4f"%np.average(subgroupAUC), "value:", "%.4f"%(weight*np.average(subgroupAUC)))
    overallAUC += weight*np.average(subgroupAUC)
print("overall AUC：", overallAUC)
print("============================================ end work ========================================================")
print("=========================================== begin work =======================================================")
