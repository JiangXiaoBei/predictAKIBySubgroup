import os
import sys
import traceback
import pandas as pd

from public import *
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import GradientBoostingClassifier

def beginWork(home, subgroupName, logger):
    logger.info("========== get subgroup auc ==========".center(CONSTANT.logLength, "="))
    constant = CONSTANT(home)
    baseAUC = pickle.load(constant.getBaseModelMetricsPath())['roc_auc']

    subgroupsPath = os.path.join(home, subgroupName, "subgroups.pkl")
    subgroups = pickle.load(subgroupsPath)
    logger.info("load subgroup in {0}".format(subgroupsPath))

    dataPath = constant.getDataPath()
    data = pd.read_pickle(dataPath).values
    allDataSize = len(data)
    akiSize, noAKISize = np.sum(data[:, -1]==1), np.num(data[:, -1]==0)
    akiRate, noAKIRate = float(akiSize)/float(size), float(noAKISize)/float(size)
    logger.info("load data in {0}".format(dataPath))
    logger.info("size:{0}, AKI:{1}({2}), NOAKI:{3}({4}), baseAUC:{5}"
        .format(data.shape[0], akiSize, akiRate, noAKISize, noAKIRate, baseAUC))

    overallAUC, subgroupAUCs, subgroupWeights = 0.0, [], []
    kfold3 = KFold(n_splits=3)
    logging.info("{0}{1}{2}{3}{4}{5}"
        .format("subgroupName".center(18), "size(rate)".center(14), "NOAKI/AKI".center(14), 
                "AKI rate".center(14), "weight".center(10), "auc".center(8)))
    gbdtParams = {
        "n_estimators": [100, 300],
        "learning_rate": [0.1, 0.5, 0.8]
    }
    treeParams = {
        "n_estimators": grid.best_params_["n_estimators"],
        "learning_rate": grid.best_params_["learning_rate"],
        "max_depth": [10, 50],
        "min_samples_split": [5, 10]
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
        AKIRate, weight = AKI/size, float(size)/float(allDataSize)
        auc = str(secondGrid.scorer_['roc_auc'])
        logger.info("{0}{1}{2}{3}{4}{5}"
            .format(subgroupName.center(18), sizeInfo.center(14), noaki_aki.center(14), 
                    str(AKIRate).center(14), str(weight).center(10), str(auc).center(8)))
        overallAUC += auc*weight

    logger.info("all subgroups are modeled, the overall auc is:{0}".format(overallAUC))
    logger.info("======================================".center(CONSTANT.logLength, "="))

# home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
# home = "/home/xinhos/PythonProjects"
if __name__ == "__main__":
    home = "/home/huxinhou/WorkSpace_XH"
    constant = CONSTANT(home)
    subgroupName = sys.argv[0]
    curSubgroupsDir = os.path.join(home, constant.getSavedDataDir(),subgroupName)
    logFilePath = os.path.join(curSubgroupsDir, "getSubgroupAUC.log")
    allLogPath = os.path.join(curSubgroupsDir, "allLog.log")
    logger = getLogger(logFilePath, allLogPath)

    try:
        beginWork(home, subgroupName,logger)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())