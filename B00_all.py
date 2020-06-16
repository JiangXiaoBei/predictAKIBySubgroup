import traceback
import numpy as np
import pandas as pd
import pygame  # pip install pygame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from concurrent.futures import ThreadPoolExecutor
from public import *

class TreeNode:
    """决策树节点对象（新建一个节点的时候作为一个叶子节点）"""
    def __init__(self, samplesIndex, auc, depth):
        self.samplesIndex = samplesIndex # 当前节点所拥有的全部节点（下标）
        self.isLeaf = True               # 当前节点是否为叶子节点
        self.akiSize = None              # 当前节点中AKI数量
        self.noAkiSize = None            # 当前节点中NOAKI的数量
        self.akiRate = None              # 当前节点的AKI率
        self.auc = auc                   # 当前节点的AUC
        self.index = -1                  # 当前节点的序号
        self.curDepth = depth            # 当前节点所处深度
        self.featurePath = []            # 从根节点到达当前节点的全部（划分）特征
        self.splitIndex = -1             # 分裂当前节点所用特征的下标   （内部节点有效）
        self.splitValue = None           # 分裂当前节点的特征取值      （内部节点，且分割特征为离散变量有效）
        self.isDispersed = False         # 分割特征是否为离散变量      （内部节点有效）
        self.subNodes = {}               # 当前节点的子节点           （内部节点有效）
        self.label = None                # 当前节点的label           （叶子节点有效）
        self.parent = -1
        
        self.sampleSize = len(self.samplesIndex)              # 当前节点中的样本数量 
        self.isLeaf = True if self.splitIndex >= 0 else False

class SDecision:
    """
    【注】在全部代码中，传递的node都只是样本的下标，真正的样本数据保存在self.samples中，传递的特征也是Dataframe中的列名
    """
    def __init__(self):
        self.__counter = 1
        self.samples = None
        self.processedsamples = None
        self.unchangeableFeatureNames = []
        self.minSamplesLeaf = 200
        self.minSampleSplit = 400
        self.maxDepth = 4
        self.overallAuc = 0.0
        self.sampleSize = 0
        self.tree = None
        self.trainFeatureSize = 1000        # 参与训练的特征数量（进行特征选择） 

    def predictSample(x, root):
        """预测节点x的类别（递归匹配节点）"""
        if root.isLeaf:
            return root.label, root.index

        curValue = x[root.splitIndex]
        subNodeKey = getSubNodeKeyBy(curValue, root.splitValue)
        subNode = root.subNodes[subNodeKey]        
        return predictSample(x, subNode)

    def predict(X):
        """预测X，返回预测值以及所属叶子节点下标"""
        Y, indexs = []
        for x in X:
            y, index = self.__predictSample(x)
            Y.append(y)
            indexs.append(index)
        return np.array(Y), np.array(indexs)

    def fitBySamples(self, samples, auc=1):
        self.samples = samples
        self.sampleSize = samples.values.shape[0]
        logger.info("{}".format("dummy the feature...".center(50, "=")))
        # 保存哑变量化后的数据
        self.processedsamples = self.samples.copy(deep=True)
        appendData, markedCols = pd.DataFrame(), []
        counter, allLen = 0, len(self.processedsamples.columns)
        for column in self.processedsamples.columns:
            if self.processedsamples[column].dtype == "object":
                newCols = pd.get_dummies(self.processedsamples[column], prefix=column, prefix_sep='_')
                appendData = pd.concat([appendData, newCols], axis=1)
                markedCols.append(column)
                counter += 1
                if counter%500 == 0:
                    logger.info("dummy the feature:{}/{}".format(counter, allLen))
        logger.info("over:{}/{}".format(counter, allLen).center(50, "="))
        self.processedsamples.drop(markedCols, axis=1, inplace=True)
        self.processedsamples = pd.concat([appendData, self.processedsamples], axis=1)
        self.unchangeableFeatureNames = self.__getFilteredUnchangeableFeature(self.samples)

        # 创建树
        depth = 1
        sampleIndexs = [i for i in range(0, len(self.samples))]
        root = TreeNode(sampleIndexs, auc, depth)
        logger.info("{}".format("begin createTree".center(50, "=")))
        self.tree = self.__createTreeFrom(root, 0)
        return self.tree

    def __getFilteredUnchangeableFeature(self, samples):
        """过滤出不可变特征名（如果划分后节点中只有1个类别或者存在某个类别）"""
        featureNames = samples.columns
        result = []
        for name in featureNames:
            prefix = name[: 3]
            if prefix=="DEM" or prefix=="VIT" or prefix=="CCS":
                groupResult = samples.groupby(name)
                if len(groupResult.size())==1:
                    continue
                result.append(name)
        return result # TODO 为了测验，只取前20条

    def __createTreeFrom(self, node, parent):
        """由传入的节点产生一棵树（判断节点是否可以分割，如果可以分割则执行分割，否则为叶子节点（新建一个Treenode时默认是叶子节点））"""
        samples = self.samples.iloc[node.samplesIndex].values
        node.index = self.__getIndex()
        node.akiSize = np.sum(samples[:, -1])
        node.noAkiSize = node.sampleSize - node.akiSize
        node.akiRate = float(node.akiSize) / float(node.sampleSize)
        node.label = 1 if np.sum(samples[:, -1]==1) > np.sum(samples[:, -1]==0) else 0
        node.parent = parent

        splitable = False
        splitIndex, splitValue, subgroupInfoList = None, None, []
        if self.__splitable(node):
            splitIndex, splitValue, subgroupInfoList = self.__doBestSplit(node)
            splitable = False if (splitValue==None)or(len(subgroupInfoList)<2) else True
        if splitable: 
            node.isLeaf = False
            node.isDispersed = True if splitValue=="" else False
            node.splitIndex = splitIndex
            node.splitValue = splitValue

            for curValue, auc, subgroupIndexs in subgroupInfoList:
                subNode = TreeNode(subgroupIndexs, auc, node.curDepth+1)
                subNode = threadPool.submit(self.__createTreeFrom, subNode, node.index).result()
                subNodeKey = self.__getSubNodeKeyBy(curValue, splitValue, isDispersed)
                node.subNodes[subNodeKey] = subNode
            logger.info("inner node {} info -- auc：{}, size：{}, parentNode：{}, subNode：{}"
                    .format(node.index, node.auc, node.sampleSize, node.parent, [sn.index for sn in node.subNodes])
            )
        else:
            self.overallAuc += float(node.sampleSize)/float(self.sampleSize)*node.auc
            logger.info("leaf node {} info -- auc：{}, size：{}, parentNode：{}".format(node.index, node.auc, node.sampleSize, node.parent))
        return node


    def __splitable(self, node):
        """
        停止分割条件：
            1. 节点属于同一类（同一类占比高于99%或多少）
            2. 节点数量小于minSampleSplit
            3. 划分后的节点数量小于minSampleLeaf
            4. 当前节点处于maxDepth
            5. 总的叶子节点数量大于maxLeafNodes
        """
        samples = self.samples.iloc[node.samplesIndex]
        notPureNode, gtMinSampleSplit, ltMaxDepth = False, False, False
        
        if node.akiSize<node.sampleSize and node.noAkiSize<node.sampleSize:
            notPureNode = True

        if node.sampleSize >= self.minSampleSplit:
            gtMinSampleSplit = True

        if node.curDepth < self.maxDepth:
            ltMaxDepth = True
        
        return notPureNode and gtMinSampleSplit and ltMaxDepth


    def __doBestSplit(self, node):
        """对node就行分割，返回所取分割特征下标、所取分割值、分割得到的子节点信息，更新节点的featurePath（subgroupInfo：curValue, auc, subgroups）"""
        usedFeaturesIndex = node.featurePath
        targetFeatureNames = list(set(self.unchangeableFeatureNames) - set(usedFeaturesIndex))
        
        maxOverallAuc = -np.inf
        splitIndex, splitValue, subgroupInfoList = None, None, []
        for featureName in targetFeatureNames:
            featureValue = self.__getBestValue(node.samplesIndex, featureName)
            result = self.__splitDataByFeature(node.samplesIndex, featureName, featureValue)
            
            if not result['splitable']:
                continue
            if result['overallAuc'] > maxOverallAuc:
                maxOverallAuc = result['overallAuc']
                splitIndex, splitValue = featureName, featureValue
                subgroupInfoList = result['subNodeInfos']
        return splitIndex, splitValue, subgroupInfoList

    def __getBestValue(self, sampleIndexs, featureName):
        """传入样本名、特征，返回该特征下划分的最佳值"""
        
        isDiscrete = isinstance(self.samples.iloc[sampleIndexs[0]][featureName], str) 
        if  isDiscrete:
            featureValues = set(self.samples.iloc[:][featureName])
            return featureValues

        # 处理连续值变量：排序后计算每对的平均值，遍历每个值，取划分后信息增益最大的那个值（参考《机器学习》-决策树章节）
        Y, T = np.array(self.samples.iloc[:]["Label"]), []
        featureValues = self.samples.iloc[:][featureName]
        sortedFeatureValues = np.sort(self.samples.iloc[:][featureName])

        D, entY, maxGain, targetValue = len(Y), self.__entropy(Y), -999999, 0
        for i in range(1, len(sortedFeatureValues)):
            t = float(sortedFeatureValues[i-1]+sortedFeatureValues[i]) / 2.0
            Dtp, Dtn = Y[featureValues>=t], Y[featureValues<t]
            if len(Dtp)==0 or len(Dtn)==0:
                continue
            gain = entY - (len(Dtp)/D*self.__entropy(Dtp) + len(Dtn)/D*self.__entropy(Dtn))
            if gain > maxGain:
                maxGain, targetValue = gain, t
        return targetValue


    
    def __splitDataByFeature(self, sampleIndexs, featureName, featureValues):
        """
        根据featureName、featureValue将样本集分割为几个子节点，对这些子节点进行建模，最终输出总auc、子节点信息

        划分的过程中，如果某个子节点的样本数小于minSamplesLeaf，则和周围的节点进行合并
        """
        result = {}
        subNodeInfos = []
        result['splitable'] = True

        # 将sampleIndexs按照featureName、featureValues分成不同的子节点
        subNodeMap = {}
        isDiscrete = isinstance(featureValues, set)
        if  isDiscrete:
            for key in featureValues:
                subNodeMap[key] = []
            for sampleIndex in sampleIndexs:
                subNodeMap[self.samples.iloc[sampleIndex][featureName]].append(sampleIndex)
            
            keys = list(subNodeMap.keys())
            while len(keys) > 2:  # 从小到大排序，从最小节点开始向上合并，一次while循环合并一次，最多最终只剩下2个节点        
                # 找出最小节点，如果大于等于minSamplesLeaf则继续寻找，否则找到次小节点，并两者合并
                minSize = self.sampleSize*10
                targetKey = 0
                for key in keys: 
                    if len(subNodeMap[key]) < minSize:
                        targetKey, minSize = key, len(subNodeMap[key])
                if minSize >= self.minSamplesLeaf:
                    break

                # 找出次小节点，并与之合并
                minSize = self.sampleSize*10
                nextTargetKey = 0
                for key in keys: # 选出次小的key
                    if key != targetKey and len(subNodeMap[key])<minSize:
                        nextTargetKey, minSize = key, len(subNodeMap[key]) 
                subNodeMap[targetKey].extend(subNodeMap[nextTargetKey])   
                subNodeMap[nextTargetKey] = subNodeMap[targetKey]
                keys.remove(targetKey)
        else:
            subNodeMap['gt'], subNodeMap['lt'] = [], []
            for sampleIndex in sampleIndexs:
                key = 'gt' if self.samples.iloc[sampleIndex][featureName] > featureValues else 'lt'
                subNodeMap[key].append(sampleIndex)

        # 对得到的子节点进行建模，并计算全部子节点的加权auc
        overallAuc = 0.0
        for subNodeIndexs in subNodeMap.values():
            if len(subNodeIndexs) < self.minSamplesLeaf:
                result['splitable'] = False
                return result

            curValue = self.samples.iloc[subNodeIndexs[0]][featureName]
            samples = self.processedsamples.iloc[subNodeIndexs].values   # 注意，这里用的是预处理后的数据
            X, Y = samples[:, :-1], samples[:, -1]
            X = SelectKBest(chi2, k=self.trainFeatureSize).fit_transform(X, Y)
            auc = round(threadPool.submit(getAUCByGBDT, X, Y).result(), 4)
            overallAuc += auc * float(len(subNodeIndexs))/float(len(sampleIndexs))
            subNodeInfos.append([curValue, auc, subNodeIndexs])                # 取出来的顺序也是：curValue, auc, subNodeIndexs

        result['subNodeInfos'] = subNodeInfos
        result['overallAuc'] = overallAuc
        return result


    def __getIndex(self):
        """获取一个节点下标（下标每次自动增长1）"""
        self.__counter += 1
        return self.__counter - 1


    def __entropy(self, Y):
        size = len(Y)
        aki = np.sum(Y == 1)
        noAki = size - aki
        akiRate, noAkiRate = float(aki)/float(size), float(noAki)/float(size)
        return -(akiRate*np.log2(akiRate) + noAkiRate*np.log2(noAkiRate))
    

    def __getSubNodeKeyBy(self, curValue, splitValue, isDispersed):
        """
        获取代表该子节点的key（在预测时需要用到，直接通过key来判断需要跳转到哪个子节点）

        如果是分割特征为离散变量，则key直接为该特征值，否则为数字类型，判断和分割值得大小取'gt'、'lt'
        """
        if isDispersed:
            key = curValue
        else:
            key = "gt" if splitValue>root.splitValue else "lt"
        return key


def getAUCByGBDT(X, Y):
    """使用gbdt建模并获取AUC"""
    if len(X) == 1:
        return 1;
    param = {
        "n_estimators": 100, 
        "learning_rate": 0.1,
        "subsample": 0.9,
        "loss": 'deviance',
        "max_depth": 10,
        "min_samples_leaf": 3,    
    }
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.3)

    clf = GradientBoostingClassifier(**param).fit(XTrain, YTrain)
    return roc_auc_score(YTest, clf.predict(XTest)) 

def toDot(tree):
    pass


def showLeaf(tree):
    if tree.isLeaf:
        logger.info("leaf size:{}, auc:{}, akiRate:{}".format(
            str(tree.sampleSize).center(6), 
            str(round(tree.auc, 4)).center(6), 
            str(round(tree.akiRate, 4)).center(6)
        ))
    for subTree in tree.subNodes.values():
        showTree(subTree)

# 貌似只能播放单声道音乐，可能是pygame模块限制
def playMusic(filename, loops=0, start=0.0, value=1.0):
    """
    :param filename: 文件名
    :param loops: 循环次数
    :param start: 从多少秒开始播放
    :param value: 设置播放的音量，音量value的范围为0.0到1.0
    :return:
    """
    flag = False  # 是否播放过
    pygame.mixer.init()  # 音乐模块初始化
    while 1:
        if flag == 0:
            pygame.mixer.music.load(filename)
            # pygame.mixer.music.play(loops=0, start=0.0) loops和start分别代表重复的次数和开始播放的位置。
            pygame.mixer.music.play(loops=loops, start=start)
            pygame.mixer.music.set_volume(value)  # 来设置播放的音量，音量value的范围为0.0到1.0。
        if pygame.mixer.music.get_busy() == True:
            flag = True
        else:
            if flag:
                pygame.mixer.music.stop()  # 停止播放
                break


threadPool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="thread_")
home = "/home/xinhos/PythonProjects"
constant = CONSTANT(home)
logFilePath = os.path.join(constant.getLogDir(), "B00-split-tree.log")
allLogPath = os.path.join(constant.getLogDir(), "B00-allLog.log")
logger = getLogger(logFilePath, allLogPath)
if __name__ == "__main__":
    try:
        np.seterr(invalid='ignore')
        samples = pd.read_pickle("/home/xinhos/PythonProjects/predictAKIBySubgroup/savedData/filteredSamples.pkl")
        logger.info("{}".format("load data...".center(50, "=")))
        tree = SDecision()
        tree.fitBySamples(samples, 1)
        logger.info(tree.overallAuc)
    except Exception as e:
        logger.error("========== catch exception! ==========".center(CONSTANT.logLength, "="))
        logger.error(traceback.format_exc())
        playMusic('/home/xinhos/Desktop/a.mp3')
