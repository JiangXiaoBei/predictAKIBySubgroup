import os
import traceback
import pickle
import shutil

from public import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def beginWork(home, logger):
    constant = CONSTANT(home)
    logger.info("========== get subgroup ==========".center(CONSTANT.logLength, "="))
    data, featureName = loadData(constant.getDataFilteredPath(), logger)
    X, Y = data[:, :-1], data[:, -1]
    size = X.shape[0]

    subgroupSizes = [4, 8, 16]
    cartModelDir = constant.getCartModelDir()
    shutil.rmtree(cartModelDir)
    os.makedirs(cartModelDir)
    for subgroupSize in subgroupSizes:
        curCartModelDir = os.path.join(cartModelDir, "subgroup-"+str(subgroupSize))
        os.makedirs(curCartModelDir)
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
        itemIndex = clf.apply(X)
        for sampleIndex, groupIndex in enumerate(itemIndex):
            if subgroups.get(groupIndex) is None:
                subgroups[groupIndex] = []
            subgroups[groupIndex].append(sampleIndex)
        with open(subgroupsSavedPath, "wb") as file:
            pickle.dump(subgroups, file)
        logger.info("the number of subgroup:{}".format(len(subgroups)))
        logger.info("subgroup data saved in {}".format(subgroupsSavedPath))
        logger.info("{0}{1}".format("subgroup index".center(20), "subgroup size".format(20)))
        for subgroupName, subgroup in subgroups.items():
            logger.info("{0}{1}".format(str(subgroupName).center(20), str(len(subgroup)).center(20)))
    logger.info("==================================".center(CONSTANT.logLength, "="))

# home = "/panfs/pfs.local/work/liu/xzhang_sta/huxinhou"
# home = "/home/xinhos/PythonProjects"
if __name__ == "__main__":
    home = "/home/huxinhou/WorkSpace_XH"
    constant = CONSTANT(home)
    logFilePath = os.path.join(constant.getLogDir(), "03-getSubgroup.log")
    allLogPath = os.path.join(constant.getLogDir(), "00-allLog.log")
    logger = getLogger(logFilePath, allLogPath)
    try:
        beginWork(home, logger)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())