import pickle as pickle

with open("negative.txt", "rb") as f:
    dictname = pickle.load(f)

print(dictname)
