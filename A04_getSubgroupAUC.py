import os
import sys
import traceback
import pandas as pd

from public import *
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

def beginWork(home, subgroupName, logger):
    logger.info("========== get subgroup auc ==========".center(CONSTANT.logLength, "="))
    constant = CONSTANT(home)

    subgroupsPath = os.path.join(home, constant.getCartModelDir(), subgroupName, "subgroups.pkl")
    with open(subgroupsPath, "rb") as file:
        subgroups = pickle.load(file)
    logger.info("load subgroup in {0}".format(subgroupsPath))

    dataPath = constant.getDataFilteredPath()
    data, _ = loadData(dataPath, logger)
    allDataSize = len(data)
    akiSize, noAKISize = np.sum(data[:, -1]==1), np.sum(data[:, -1]==0)
    akiRate, noAKIRate = float(akiSize)/float(allDataSize), float(noAKISize)/float(allDataSize)
    logger.info("load data in {0}".format(dataPath))
    logger.info("size:{0}, AKI:{1}({2}), NOAKI:{3}({4})"
        .format(data.shape[0], akiSize, round(akiRate, 4), noAKISize, round(noAKIRate, 4)))

    overallAUC, subgroupAUCs, subgroupWeights = 0.0, [], []
    kfold3 = KFold(n_splits=3)
    logger.info("{0}{1}{2}{3}{4}{5}"
        .format("subgroupName".center(18), "size(rate)".center(14), "NOAKI/AKI".center(14), 
                "AKI rate".center(14), "weight".center(10), "auc".center(8)))
    
    gbdtParams = {
        "n_estimators": [20, 50, 100],
        "learning_rate": [0.05, 0.1, 0.5]
    }
    treeParams = {
        "max_depth": [3, 5, 10, 50],
        "min_samples_split": [2, 5, 10]
    }
    for subgroupName, subgroupItemIndex in subgroups.items():
        # 使用网格搜索，3折交叉验证
        subgroupData = data[np.array(subgroupItemIndex)]
        X, Y = subgroupData[:, :-1], subgroupData[:, -1]
        firstGrid, secondGrid = GBDT(X, Y, gbdtParams, treeParams, 3, logger)
        
        size = len(subgroupData)
        sizeInfo = str(size) + "(" + str(float(size)/float(allDataSize)) + ")"
        NOAKI, AKI = np.sum(subgroupData[:, -1]==0), np.sum(subgroupData[:, -1]==1)
        noaki_aki = str(NOAKI) + "/" + str(AKI)
        AKIRate, weight = round(AKI/size, 4), round(float(size)/float(allDataSize), 4)
        auc = round(secondGrid.cv_results_['mean_test_roc_auc'][secondGrid.best_index_], 4)
        logger.info("{0}{1}{2}{3}{4}{5}"
            .format(str(subgroupName).center(18), sizeInfo.center(14), noaki_aki.center(14), 
                    str(AKIRate).center(14), str(weight).center(10), str(auc).center(8)))

        overallAUC += auc*weight

    logger.info("all subgroups are modeled, the overall auc is:{0}".format(overallAUC))
    logger.info("======================================".center(CONSTANT.logLength, "="))

# home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
# home = "/home/xinhos/PythonProjects"
if __name__ == "__main__":
    home = "/home/huxinhou/WorkSpace_XH"
    constant = CONSTANT(home)
    subgroupName = sys.argv[1]
    curSubgroupsDir = os.path.join(home, constant.getCartModelDir(), subgroupName)
    logFilePath = os.path.join(curSubgroupsDir, "getSubgroupAUC.log")
    allLogPath = os.path.join(curSubgroupsDir, "allLog.log")
    logger = getLogger(logFilePath, allLogPath)

    try:
        beginWork(home, subgroupName,logger)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())
