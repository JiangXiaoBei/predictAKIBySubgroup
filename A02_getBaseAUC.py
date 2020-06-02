import os
import pickle
import traceback

from public import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

"""
加载保存好的全部data数据，使用网格搜索、10折交叉验证的方式获取最佳AUC
"""
def beginWork(home, logger):
    constant = CONSTANT(home)
    dataDir = constant.getDataPath()
    baseModelPath = constant.getBaseModelPath()
    baseModelMetricsPath = constant.getBaseModelMetricsPath()
    logger.info("========== get base auc ==========".center(CONSTANT.logLength, "="))
    
    data = loadData(dataDir, logger)
    X, Y = data[:, :-1], data[:, -1]

    gbdtParams = {
        # "n_estimators": [200, 300， 400],
        # "learning_rate": [0.05, 0.1, 0.2]
    }
    treeParams = {
        # "max_depth": [10, 50],
        # "min_samples_split": [5, 10]
    }
    firstGrid, secondGrid = GBDT(X, Y, gbdtParams, treeParams, 3, logger) # TODO 测试代码正确性，训练时cv改回10

    logger.info("first stage GBDT(with gbdt param) model info:")
    showGridMetrics(firstGrid, "GBDT", logger)
    logger.info("second stage(with tree params) GBDT model info:")
    showGridMetrics(secondGrid, "GBDT(best tree)", logger)
    with open(baseModelMetricsPath, "w") as file:
        pickle.dump(secondGrid.scorer_, file)
    with open(baseModelPath, "w") as file:
        pickle.dump(secondGrid.best_estimator_, file)
    logger.info("base model saved in {0}".format(baseModelPath))
    logger.info("metrics info of base model saved in {0}".format(baseModelMetricsPath))
    logger.info("==================================".center(CONSTANT.logLength, "="))

# home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
# home = "/home/xinhos/PythonProjects"
# home = "/home/huxinhou/WorkSpace_XH"
if __name__ == "__main__":
    home = "/home/huxinhou/WorkSpace_XH"
    constant = CONSTANT(home)
    logFilePath = os.path.join(constant.getLogDir(), "02-getBaseAUC.log")
    allLogPath = os.path.join(constant.getLogDir(), "00-allLog.log")
    logger = getLogger(logFilePath, allLogPath)
    try:
        beginWork(home, logger)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())

"""
笔记：
    网格搜索时使用多个参数，返回的是make_scores()。如果需要使用的话，还需传入预测值、实际值
"""