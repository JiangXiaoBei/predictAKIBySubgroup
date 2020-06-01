import os
import traceback
import pickle

from public import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def beginWork(home, logger):
    constant = CONSTANT(home)
    logger.info("========== get subgroup ==========".center(CONSTANT.logLength, "="))
    data = loadData(constant.getDataFilteredPath(), logger)
    X, Y = data[:, :-1], data[:, -1]
    size = X.shape[0]

    subgroupSizes = [8, 16, 32, 64]
    cartModelDir = constant.getCartModelDir()
    for subgroupSize in subgroupSizes:
        curCartModelDir = os.path.join(cartModelDir, "subgroup-"+str(subgroupSize))
        if not os.path.exists(curCartModelDir):
            os.mkdir(curCartModelDir)
        dotFilePath = os.path.join(curCartModelDir, "dot.dot")
        paramsTxtPath = os.path.join(curCartModelDir, "params.txt")
        subgroupsSavedPath = os.path.join(curCartModelDir, "subgroups.pkl")
        
        subgroups = {}
        avgSize = size // subgroupSize
        param = {
            "criterion": "gini",
            "min_samples_leaf": avgSize*2 // 3,
            "min_samples_split": avgSize*2
        }
        clf = DecisionTreeClassifier(**param).fit(X, Y)

        # 保存树结构&超参数
        with open(dotFilePath, "w", encoding="utf-8") as file:
            file = export_graphviz(clf, out_file = file, feature_names = featureName, 
                    filled = True, rounded = True, special_characters = True)
        logger.info("cart structure saved in {}".format(dotFilePath))
        with open(paramsTxtPath, "w", encoding='utf-8') as file:
            file.write(str(clf.get_params()))
        logger.info("cart params saved in {}".format(paramsTxtPath))
        logger.info(param)
                            
        # 获得并保存亚组
        subgroupIndex = clf.apply(X)
        for sampleIndex, groupIndex in enumerate(subgroupIndex):
            if subgroups.get(groupIndex) is None:
                subgroups[groupIndex] = []
            subgroups[groupIndex].append(sampleIndex)
        with open(subgroupsSavedPath, "w") as file:
            pickle.dump(subgroupsSavedPath, file)
        logger.info("subgroup data saved in {}".format(subgroupsSavedPath))
        logger.info("the number of subgroup:{}".format(len(subgroups)))
        logger.info("{0}{1}".format("subgroup index".center(15), "subgroup size".format(15)))
        for subgroupName, subgroupSize in subgroups.items():
            logger.info("{0}{1}".format(str(subgroupName).center(15), str(subgroupSize).center(15)))
    logger.info("==================================".center(CONSTANT.logLength, "="))

# home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
# home = "/home/xinhos/PythonProjects"
if __name__ == "__main__":
    home = "../../"
    constant = CONSTANT(home)
    logFilePath = os.path.join(constant.getLogDir(), "03-getSubgroup.log")
    allLogPath = os.path.join(constant.getLogDir(), "00-allLog.log")
    logger = getLogger(logFilePath, allLogPath)
    try:
        beginWork(home, logger)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())