import os
import pickle
import logging
import traceback

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from public import *


def beginWork(home, logger):
    logger.info("========== load and handle data ==========".center(CONSTANT.logLength, "="))
    constant = CONSTANT(home)
    listDataDir = constant.getListDataDir()
    logDir = constant.getLogDir()

    dataPath = constant.getDataPath() 
    dataFilteredPath = constant.getDataFilteredPath()

    data, dataFiltered = None, None
    if os.path.exists(dataPath) and os.path.exists(dataFilteredPath):
        logger.info("data exist!")
        with open(dataPath, "rb") as file:
            data = pickle.load(file)
        with open(dataFilteredPath, "rb") as file:
            dataFiltered = pickle.load(file)
    else:
        dataProccessor = DataProccessor(listDataDir)

        data = dataProccessor.getAKIData(deleteEmptyCol=True)
        dataFiltered = dataProccessor.getFilteredData(deleteEmptyCol=True)

        logger.info("data saving...")
        data.to_pickle(dataPath)
        dataFiltered.to_pickle(dataFilteredPath)
        logger.info("data saved in {}".format(dataPath))
        logger.info("dataFiltered saved in {}".format(dataFilteredPath))
    showDatasInfo(data.values, dataFiltered.values, logger)
    logger.info("=======================================".center(CONSTANT.logLength, "="))

# home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
#　home = "/home/xinhos/PythonProjects"
if __name__ == "__main__":
    home = "../../"
    constant = CONSTANT(home)
    logFilePath = os.path.join(constant.getLogDir(), "01-loadAndSave.log")
    allLogPath = os.path.join(constant.getLogDir(), "00-allLog.log")
    logger = getLogger(logFilePath, allLogPath)
    try:
        beginWork(home, logger)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())