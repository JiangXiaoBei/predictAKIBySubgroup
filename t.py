import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

class DataLoader:
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
    
    
    def loadRawAKIData(self):
        if self.__rawAKIData == None:
            self.__initRawAKIData()
        return self.__rawAKIData
    
    
    def loadAKIData(self, time=1):
        """获取AKI数据（默认为发病前1天，数据中出现NAN、INF的记录被删除）"""
        if self.__AKIData is None:
            self.__initRawAKIData()
            AKIDataArray = [self.__getPreprocessedDataFrom(aRawData, time) for aRawData in self.__rawAKIData.values]
            AKIFeatureName = self.getAKIFeatureName()
            self.__AKIData = pd.DataFrame(AKIDataArray).replace([np.inf, -np.inf], np.nan).dropna(axis=0) # 删除空值
            self.__AKIData.columns = AKIFeatureName
        return self.__AKIData
    
    def getFilteredData(self):
        """获取仅包含不可变特征的数据"""
        if self.__filteredAKIData is None:
            inconpFeatureName = self.getIncompFeatureName()
            self.__filteredAKIData = self.getAKIData()[inconpFeatureName]
        return self.__filteredAKIData
    
    
    def getAKIFeatureName(self):
        """获取（预置）AKI特征名"""
        if self.__AKIFeatureName == None:
            self.__initAKIFeatureName()
        return self.__AKIFeatureName
    
    
    def getIncompFeatureName(self):
        """获取（预置）不可变特征名"""
        if self.__incompAKIFeatureName == None:
            self.__initIncompAKIFeature()
        return self.__incompAKIFeatureName
    

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
    
    def __getClassLabel(self, c):
        """有些离散值记录一会是数值类型，一会是字符串，统一格式"""
        table = {
            "0":"00", "1":"01", "2":"02", "3":"03", 
            "4":"04", "5":"05", "6":"06", "7":"07", "8": "08",
            0.0:"00", 1.0:"01", 2.0:"02", 3.0:"03",
            4.0:"04", 5.0:"05", 6.0:"06", 7.0:"07", 8.0:"08",
            "00":"00", "01":"01", "02":"02", "03":"03", 
            "04":"04", "05":"05", "06":"06", "07":"07", "08": "08"
        }
        return table[c]
    
    def __getPreprocessedDataFrom(self, aPatientRawData, time):
        """预处理数据"""
        # 获取发病日期。([6]表示提取第7个大类特征，[0]表示第1条记录)
        label, firstAKITime = aPatientRawData.T[6][0] 
        firstAKITime = float(firstAKITime) - time # 因为要提前一天预测，因此减去1
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
                     recode = ['0', 0]                   # 当没有时间记录，说明患者在这一项上为空。直接赋0
                value, time = recode 
                time = float(time)
                timeInterval = firstAKITime - time
                if timeInterval>0 and timeInterval<minTimeInterval:
                    minTimeInterval = timeInterval
                    if index==3 or index==4 or index==5: # 将Smoking_Tabacco_TabaccoType保留为离散变量
                        targetValue = self.__getClassLabel(value)
                    else:
                        targetValue = float(value)
            vitalData.append(targetValue)
        ######### 补丁代码，不知为何（可能是vital格式的原因），会有一些index==3的记录无法进入对应的分支中###########
        vitalData[3], vitalData[4], vitalData[5] = self.__getClassLabel(vitalData[3]), self.__getClassLabel(vitalData[4]), self.__getClassLabel(vitalData[5])
            
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
        ccsData = ['0' for i in range(0, self.__ccsSubFeatureNum)]
        for subFeatureData in comorbidity:
            isEmptyData = (subFeatureData == [['0']])
            if isEmptyData:
                break
            subFeatureName = subFeatureData[0][0]
            subFeatureIndex = int(subFeatureName.split("ccs")[1])
            ccsData[subFeatureIndex-1] = '1'  # 这里采用第一种处理方式：只要有记录，则标记为发过病

        # 抽取过程（procedure）数据（3维）：PX1_PX2_PX3_
        pxData = ['0' for i in range(0, self.__pxSubFeatureNum)]
        for subFeatureData in procedure:
            isEmptyData = (subFeatureData == [['0']])
            if isEmptyData:
                break
            subFeatureName = subFeatureData[0][0]
            subFeatureIndex = int(subFeatureName.split("px")[1])
            pxData[subFeatureIndex-1] = '1'  # 这里采用第一种处理方式：只要有记录，则标记为发过病

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

def deleteNullCol(data):
    markedIndex = []
    for column in data.columns[: -2]:
        if (data[column].nunique() == 1):
            markedIndex.append(column)
    return data.drop(markedIndex, axis=1, inplace=False)

home = "/home/xinhos/PythonProjects/predictAKIBySubgroup"
AKIDir = os.path.join(home, "listData")
dataLoader = DataLoader(AKIDir)
samples = dataLoader.loadAKIData()

inconpFeatureName = dataLoader.getIncompFeatureName()
filteredSamples = dataLoader.loadAKIData()[inconpFeatureName]
filteredSamples
filteredSamples.to_pickle("/home/xinhos/PythonProjects/predictAKIBySubgroup/savedData/ffff.pkl")