# -*- coding: utf-8 -*-
"""
Spyder Editor

This script contains useful functions to evaluate sequences generated by the generator. 
"""

import dna
import itertools
import pandas as pd
import numpy as np

#used this variables for testing. will be removed.
sequences = dna.load_dna_data(13000,1000,"../Data",["Human"],1)
train_sequences = sequences[0][0]
test_sequences = sequences[1][0]

def decode_dna_matrices(sequences):
    seq_shape =sequences[0].shape 
    decoded_sequences = np.empty([sequences.shape[0],seq_shape[1]],dtype='object')
    for ind,seq in enumerate(sequences):
        values = np.zeros([seq_shape[0],seq_shape[1]])
        for i in range(seq_shape[0]):
            values[i,:] = seq[i].reshape(seq_shape[1])
        values_df = pd.DataFrame(values).T
        values_df.columns = ['A','C','T','G']
        x=values_df.stack()
        decoded_sequence=x[x!=0].index.get_level_values(1).tolist()
        decoded_sequences [ind,:] = decoded_sequence
    return (decoded_sequences)


##Computing edit distances 
#from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def edit_dist(s1,s2):
    if len(s1) < len(s2):
        return edit_dist(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
    
def pairwise_edit_distances (train_sequences,test_sequences):
    print ("Decoding seq 1")
    dec_tr_seq = decode_dna_matrices(train_sequences)
    print ("Decoding seq 2")
    dec_tst_seq = decode_dna_matrices(test_sequences)
    nr_tr= train_sequences.shape[0]
    nr_tst=test_sequences.shape[0]
    edit_distances = np.zeros(nr_tr*nr_tst)
    n_id = 0
    print ("Computing distances")
    for tr_ind,tst_ind in itertools.product(list(range(nr_tr)), list(range(nr_tst))):
        edit_distances[n_id] = edit_dist(dec_tr_seq[tr_ind,:],dec_tst_seq[tst_ind,:])
        n_id=n_id+1
    return(edit_distances)
        
        
    