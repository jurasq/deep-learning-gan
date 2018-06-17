# -*- coding: utf-8 -*-
"""
Spyder Editor

Analyse generated sequences - script
"""
import dna
import evaluateSequences
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
from multiprocessing.pool import ThreadPool
#First analyze the distance to a set of testing sets
#sequences = dna.load_dna_data(100,1000,"../Data",["Human"],"pos")
#seq_subset = sequences[0][0]

#seq_subset1 = seq_subset[0:15]
#seq_subset2 = seq_subset[51:100]

#first compute distances between sequences in training set
#dist_train=evaluateSequences.pairwise_edit_distances (seq_subset1,seq_subset2)

#positive or negative?
seq_dir = sys.argv[1]
#should be something like this '../TripleGAN_dna_50_200/TripleGAN_generated_sequences_epochnegative_'
outputName=sys.argv[2]
#should be somehing like "negative"

epfiles = ['../TripleGAN_dna_50_200/'+filename for filename in os.listdir('../TripleGAN_dna_50_200/') if filename.startswith(seq_dir)]
epfiles=np.sort(epfiles)
print(epfiles)
dists={}
def perform_the_thing(filename):
    with open(filename, "r") as f:
        seq_array_local = []
        # decode seq to strings
        for line in f:
            # Strip the endline character from each input line
            line = line.rstrip("\n")
            seq_array_local.append(list(line))
        print('Analysing ' + filename)
        seq_array_local = seq_array_local[0:50]
    dists[filename] = evaluateSequences.pairwise_edit_distances(np.asarray(seq_array_local),np.asarray(seq_array_local), False)
    print("%s finished" % filename)

#compute distances between sequences every 10 epochs
#epochs = ['0'+str(i)+'0' for i in range(0,10)]
#epochs.append('100')
#epochs.append('109')
#dec_seq_subset2 = evaluateSequences.decode_dna_matrices(seq_subset2)

# for filename in epfiles:
# #    filename = +'_'+ep+'.txt'
#     fileInput = open(filename, "r")
#     seq_array=[]
# #decode seq to strings
#     for strLine in fileInput:
#         #Strip the endline character from each input line
#         strLine = strLine.rstrip("\n")
#         seq_array.append(list(strLine))
#     print('Analysing '+filename)
#     seq_array = seq_array[0:50]
#     dists.append(evaluateSequences.pairwise_edit_distances (np.asarray(seq_array),np.asarray(seq_array),False))

pool = ThreadPool(processes=5)
pool.map(perform_the_thing, [filename for filename in epfiles])
pool.close()
pool.join()

print("Done with concurrent for loop")

#save distances
output_file = outputName+".txt"
with open(output_file, "wb") as fp:   #Pickling
    pickle.dump(dists, fp)

#produce boxplot
#plt.boxplot(dists)   
