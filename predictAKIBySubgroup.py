import os
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

"""
DataProccessor：一个读取指定目录下pkl格式的AKI数据处理器。pkl数据的格式见<DataReader.py>。
    可获取rawAKIData、将大类特征全部展开后的AKIData，也可以获取仅含有不可变数据的ADIData（filteredAKIData）
    
    DataProccessor(AKIDataDir):
        创建一个DataProcessor对象，传入要读取的AKIData目录，将自动读取该目录下全部pkl文件（包括子文件中的）
        【输入】目标AKIData目录
    
    setAKIDaDir(AKIDataDir)
        设置需要读取的AKIData目录。设置后会销毁已经读取的数据，并重新载入新目录下的数据
        【从】目标AKIData目录
    
    getAKIDaDir()
        获取目标AKIData目录
        
    getRawAKIData()
        获取目标AKIData目录下的全部原生数据（多个pkl文件中的数据读取为1个）
    
    getAKIFeatureName()
        DataProccessor对象内部维护了一个featureName数组，保存了全部7大类共3w多个特征的名字
    
    getAKIData(deleteEmptyCol=False)
        获取处理后的AKIData，处理过程主要是将全部大类特征展开，将全部3w个子特征排列（不同的大类特征不要不同的处理）。
        可选是否传入一个boolean类型参数，代表是否删除数据中的空列（即全部样本的该列值都相同，对决策树预测建模没有用）
        【参数】
            deleteEmptyCol，<boolean>，是否删除数据中的空列
        【返回】
            可用于决策树训练的AKI数据
    
   getFilteredData(deleteEmptyCol=False) 
       获取只包含不可变特征的数据。
       
    【未完成工作】
        1. 获取每个大类特征/展开后的大类特征数据
"""
class DataProccessor:
    def __init__(self, AKIDataDir):
        self.__demoSubFeatureNum = 4
        self.__vitalSubFeatureNum = 8
        self.__labSubFeatureNum = 817
        self.__ccsSubFeatureNum = 2621
        self.__pxSubFeatureNum = 15606
        self.__medSubFeatureNum = 11538
        self.__incompAKIFeatureName = None
        self.__AKIFeatureName = None
        
        self.setAKIDaDir(AKIDataDir)
    
    """
    设置目标AKI数据目录。如果DataGenerator已经读取过一个目录下的数据，设置新目录后将会加载新数据
    【输入】想要读取的AKI数据目录（目录要求同创建DataGenerator对象）
    """
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
        
    
    """
    获取AKI数据（样本中的无用列被去除，无用列指的是全部样本的该列值相同）
    """
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
    
    
    """
    获取AKIData，抽取出其中不可变部分：
        1. 全部人口统计学（4，Age、Hispanic、Race、Sex）
        2. 生命体征（6，HT、WT、ORIGINAL_BMI、Smoking、TOBACCO、TOBACCO_TYPE）
        3. 全部并发症（280）【存疑，部分入院后并发症是否可以作为可改变因素？】
    """
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
            
    
    """ 
    【输入】保存所有结构化后数据的目录。预处理保存好的数据。要求AKIDataDir下只存在由<dataHelper.py>所处理而得到的pkl文件
        （处理后得到数据的格式参见dataHelper.py）
    【输出】全部结构化的数据，一张pandas.DataFrame表，包含7个大类特征（大类特征中包含子特征，但是没有分割）
    """
    def __initRawAKIData(self):
        rawAKIdata = []
        for fileName in os.listdir(self.__AKIDataDir):
            if os.path.isfile(os.path.join(self.__AKIDataDir, fileName)):
                rawAKIdata.extend(pd.read_pickle(os.path.join(self.__AKIDataDir, fileName)))
        self.__rawAKIData = pd.DataFrame(rawAKIdata)
        self.__rawAKIData.columns = ['demo', 'vital', 'lab', 'comorbidity', 'procedure', 'med', 'AKI_label']

    """
    【输入】无
    【输出】全部大类展开后的特征名，按照'demo', 'vital', 'lab', 'comorbidity', 'procedure', 'med', 'AKI_label'的顺序排列
    """
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
    
    
    """
    【输入】一个格式为[]
    设定要抽取的列表（一级列表、二级列表），将不可变特征抽取出来。
    1. 全部人口统计学（4，Age、Hispanic、Race、Sex）
    2. 生命体征（5，HT、WT、ORIGINAL_BMI、TOBACCO、TOBACCO_TYPE）
    3. 全部并发症（280）
    """ 
    def __getPreprocessedDataFrom(self, aPatientRawData):
        # 获取发病日期。([6]表示提取第7个大类特征，[0]表示第1条记录)
        label, firstAKITime = aPatientRawData.T[6][0] 
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

# ======================================================================================================================
# ==================================================开始======================================================================
# ======================================================================================================================
AKIDataDir = "/home/xinhos/Desktop/AKI_newdata1/"
dataProccessor = DataProccessor(AKIDataDir)
data = dataProccessor.getAKIData(deleteEmptyCol=True)
dataFiltered = dataProccessor.getFilteredData(deleteEmptyCol=True)

data = DataProccessor.preprocess(data)
dataFiltered = DataProccessor.preprocess(dataFiltered)
featureName = dataFiltered.columns
data, dataFiltered = data.values, dataFiltered.values

# 1. 获得基线AUC
kfold = KFold(n_splits=10)
for trainDataIndex, testDataIndex in kfold.split(data):
    trainData, testData = data[trainDataIndex, :], data[testDataIndex, :]
    trainX, trainY = trainData[:, :-2], trainData[:, -1]
    testX, testY = testData[:, :-2], testData[:, -1]
    GBDTClf = GradientBoostingClassifier().fit(trainX, trainY)
    predictedTestY = GBDTClf.predict(testX)
    auc = roc_auc_score(testY, predictedTestY)
baseAUC = np.average(aucList)

# 2. 使用CART进行亚组分组，并获取亚组
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
subgroups = {}
trainX, trainY = dataFiltered[:, :-2], dataFiltered[:, -1]

param = {
    "criterion": "gini",
    "min_samples_leaf": 50,
    "min_samples_split": 500
}
cartClf = DecisionTreeClassifier(**param)
cartClf.fit(trainX, trainY)

# 保存决策树结构
with open("./tree-struct.dot", "w", encoding="utf-8") as file:
    file = export_graphviz(cartClf, out_file = file, feature_names = featureName, 
                           filled = True, rounded = True, special_characters = True)
# 获得亚组
subgroupIndex = cartClf.apply(dataFiltered)
for sampleIndex, groupIndex in enumerate(subgroupIndex):
    if subgroups.get(groupIndex) == None:
        subgroups[groupIndex] = []
    subgroups[groupIndex].append(sampleIndex)

# 3. 在得到的每个亚组上使用GBDT建模，并计算总体AUC，以及输出AUC变化情况
overallAUC = 0.0
subgroupAUCs = []
kflod = KFold(n_splits=3)
for _, subgroup in subgroups.items():
    subgroup = np.array(subgroup)
    subgroupData = data[subgroup]
    for trainDataIndex, testDataIndex in kfold.split(subgroupData):
        trainData, testData = subgroupData[trainDataIndex, :], subgroupData[testDataIndex, :]
        trainX, trainY = trainData[:, :-2], trainData[:, -1]
        testX, testY = testData[:, :-2], testData[:, -1]
        GBDTClf = GradientBoostingClassifier().fit(trainX, trainY)
        predictedTestY = GBDTClf.predict(testX)
        auc = roc_auc_score(testY, predictedTestY)
    baseAUC = np.average(aucList)
    
# 4. 输出亚组信息&保存数据
print("==========================================================================")
print("亚组个数：", len(subgroups))
count = 0
for subgroupName, subgroup in subgroups.items():
    count += 1
    sign = ''
    if count % 5 == 0:
        sign = '\n'
    print("\t亚组" + str(subgroupName).zfill(3) + "：" + str(len(subgroup)) 
          + "auc："+str(subgroupAUCs[subgroupName]), end=sign)
print("总体AUC：", overallAUC)
print("==========================================================================")
