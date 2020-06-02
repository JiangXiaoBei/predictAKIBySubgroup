import os
import pickle
import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

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
        # 将离散变量哑变量化，TODO 注意下面4+8引入了常数，可以优化掉。
        tmpList = AKIData.columns[: 4+8]
        for column in tmpList:
            if AKIData[column].dtype == "object":
                newCols = pd.get_dummies(AKIData[column], prefix=column, prefix_sep='_')
                appendData = pd.concat([appendData, newCols], axis=1)
                markedCols.append(column)
        AKIData.drop(markedCols, axis=1, inplace=True)
        AKIData = pd.concat([appendData, AKIData], axis=1).values

        # 处理NaN、INF（只保留非nan、非inf的行）
        infIndex = np.isinf(AKIData).any(axis=1)
        nanIndex = np.isnan(AKIData).any(axis=1)
        AKIData = AKIData[~infIndex, :]
        AKIData = AKIData[~nanIndex, :]

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

class CONSTANT:
    logLength = 70

    def __init__(self, home):
        self.__projectDir = os.path.join(home, "predictAKIBySubgroup")
        self.__listDataDir = os.path.join(self.__projectDir, "listData/")
        self.__savedDataDir = os.path.join(self.__projectDir, "savedData")
        self.__logDir = os.path.join(self.__projectDir, "log")

        self.__dataPath = os.path.join(self.__savedDataDir, "data.pkl") 
        self.__dataFilteredPath = os.path.join(self.__savedDataDir, "dataFiltered.pkl")

        self.__baseModelDir = os.path.join(self.__savedDataDir, "baseModel")
        self.__baseModelPath =  os.path.join(self.__baseModelDir, "baseModel.m")
        self.__baseModelMetricsPath = os.path.join(self.__baseModelDir, "baseModelMetrics.pkl")

        self.__cartModelDir =  os.path.join(self.__savedDataDir, "subgroupModel")

    def getProjectDir(self):
        return self.__projectDir    
    def getListDataDir(self):
        return self.__listDataDir
    def getSavedDataDir(self):
        return self.__savedDataDir
    def getLogDir(self):
        return self.__logDir

    def getDataPath(self):
        return self.__dataPath
    def getDataFilteredPath(self):
        return self.__dataFilteredPath

    def getBaseModelDir(self):
        return self.__baseModelDir
    def getBaseModelPath(self):
        return self.__baseModelPath
    def getBaseModelMetricsPath(self):
        return self.__baseModelMetricsPath

    def getCartModelDir(self):
        return self.__cartModelDir

def getLogger(logFilePath, allLogPath=None):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    ALL_LOG_FORMAT = "%(asctime)s - %(thread)d - %(funcName)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(LOG_FORMAT)
    allLogFormatter = logging.Formatter(ALL_LOG_FORMAT)
    logger = logging.getLogger(__name__)#.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter(formatter)

    fileHandler = logging.FileHandler(logFilePath, "w")
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.INFO)

    if allLogPath is not None:
        fileHandler2 = logging.FileHandler(allLogPath, "a")
        fileHandler2.setFormatter(allLogFormatter)
        fileHandler2.setLevel(logging.INFO)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.addHandler(fileHandler2)
    return logger

def showDatasInfo(dataArr, dataFilteredArr, logger):
    logger.info("{0}{1}{2}".format("".rjust(5), 
                            "data".center(17), 
                            "dataFiltered".center(17)))
    logger.info("{0}{1}{2}".format("size".ljust(5), 
                            str(dataArr.shape).center(17), 
                            str(dataFilteredArr.shape).center(17)))
    logger.info("{0}{1}{2}".format("AKI".ljust(5), 
                            str(np.sum(dataArr[:, -1]==1)).center(17), 
                            str(np.sum(dataFilteredArr[:, -1]==1)).center(17)))
    logger.info("{0}{1}{2}".format("NOAKI".ljust(5), 
                            str(np.sum(dataArr[:, -1]==0)).center(17), 
                            str(np.sum(dataFilteredArr[:, -1]==0)).center(17)))

def showDataInfo(dataArr, logger):
    logger.info("{0}{1}".format("size".ljust(5), str(dataArr.shape).center(17)))
    logger.info("{0}{1}".format("AKI".ljust(5), str(np.sum(dataArr[:, -1]==1)).center(17)))
    logger.info("{0}{1}".format("NOAKI".ljust(5), str(np.sum(dataArr[:, -1]==0)).center(17)))

def showGridMetrics(grid, modelName, logger):
    bestIndex = grid.best_index_
    results = grid.cv_results_
    bestAuc = grid.best_score_
    bestGbdtParams = grid.best_params_

    logger.info("best {0} parameter:{1}".format(modelName, bestGbdtParams))
    logger.info("auc:{0}, f1:{1}, accuracy:{2}, precition:{3}, recall:{4}".format(
        round(results['mean_test_roc_auc'][bestIndex], 4), 
        round(results['mean_test_f1'][bestIndex], 4),
        round(results['mean_test_accuracy'][bestIndex], 4),
        round(results['mean_test_precision'][bestIndex], 4),
        round(results['mean_test_recall'][bestIndex], 4)
    ))

def GBDT(X, Y, gbdtParams, treeParams, cv, logger):
    logger.info("training model with GBDTParams by GridSearchCV")
    firstGrid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=gbdtParams, refit="roc_auc",
                n_jobs=-1, scoring=['roc_auc', 'f1', 'accuracy', 'precision', 'recall'], cv=cv)
    firstGrid.fit(X, Y)

    treeParams.update(gbdtParams)
    logger.info("training model with treeParams by GridSearchCV")
    secondGrid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=treeParams, refit="roc_auc",
                n_jobs=-1, scoring=['roc_auc', 'f1', 'accuracy', 'precision', 'recall'], cv=cv)
    secondGrid.fit(X, Y)

    return firstGrid, secondGrid

def loadData(dataPath, logger):
    logger.info("load data in{}".format(dataPath))
    data = pd.read_pickle(dataPath)
    data = DataProccessor.preprocess(data)
    showDataInfo(data, logger)
    return data


"""
笔记：
    网格搜索时使用多个参数，
"""