from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 用于分类的输入向量是inX，输入的训练样本集为dataSet
# 标签向量为 labels ，最后的参数 k 表示用于选择最近邻居的数目，其中标签向量的元素数目和矩阵 dataSet 的行数相同。
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet         # tile 代表了inX，复制为dataSetSize行，1列的数组
    sqDiffMat = diffMat**2                                  # 平方
    sqDistances = sqDiffMat.sum(axis=1)                     # axis 等于 1 是将 矩阵的每一行 相加
    distances = sqDistances**0.5                            # 开方
    sortedDistIndices = distances.argsort()  # 从小到大 排列，argsort：将distacnces中的元素从小到大排列，提取其对应的index(索引)然后输出到sortedDistances
    classCount = {}                                         # 求出来最低距离的 labels结果，存放在classCount 中
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 将文本记录转换Numpy的解析程序
def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberofLines = len(arrayOLines)                # 得到文件行数
    returnMat = zeros((numberofLines, 3))           # 创建返回的Numpy矩阵
    classLabelVector = []                           # 解析文件数据到列表
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector




