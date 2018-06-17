# -*- coding: utf-8 -*-
"""
Spyder Editor

Analyse generated sequences - script
"""
import dna
import evaluateSequences
import numpy as np
import matplotlib.pyplot as plt
#First analyze the distance to a set of testing sets
sequences = dna.load_dna_data(100,1000,"../Data",["Human"],"pos")
seq_subset = sequences[0][0]

seq_subset1 = seq_subset[0:15]
seq_subset2 = seq_subset[51:100]

#first compute distances between sequences in training set
dist_train=evaluateSequences.pairwise_edit_distances (seq_subset1,seq_subset2)

#compute distances between sequences every 10 epochs
epochs = ['0'+str(i)+'0' for i in range(0,10)]
epochs.append('100')
epochs.append('109')
dec_seq_subset2 = evaluateSequences.decode_dna_matrices(seq_subset2)
dists=[]
for ep in epochs:
    filename = '../../TripleGAN_dna_50_200/TripleGAN_generated_sequences_epochpositive_'+ep+'.txt'
    fileInput = open(filename, "r")
    seq_array=[]
#decode seq to strings
    for strLine in fileInput:
        #Strip the endline character from each input line
        strLine = strLine.rstrip("\n")
        seq_array.append(list(strLine))
    print('Analysing epoch '+str(ep))    
    seq_array = seq_array[0:50]
    dists.append(evaluateSequences.pairwise_edit_distances (np.asarray(seq_array),np.asarray(seq_array),False))
        
#produce boxplot
 plt.boxplot(dists)   