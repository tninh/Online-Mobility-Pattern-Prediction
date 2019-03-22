from sklearn import svm
import pickle

data = {"data":[], "target":[]}
with open("data1", "rb") as d, open("target1", "rb") as t:
    data["data"] = pickle.load(d)
    data["target"] = pickle.load(t)

#print(type(data["data"]))
#print(data["data"][2])
for y in data["data"]:
    #print(y)
    [float(i) for i in y]
[float(i) for i in data["target"]]
#print(data["data"])
#print(data["target"])
#print(type(data["data"][0]))

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(data["data"][:-1], data["target"][:-1])

clf.predict(data["data"][-1:])

print("Hello")