import kNN
import matplotlib
import matplotlib.pyplot as plt
group, labels = kNN.createDataSet()
print(group)
print(labels)
print(kNN.classify0([0, 0], group, labels, 3))
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')
print(datingDataMat)
print(datingLabels[0:20])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show()