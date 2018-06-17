# -*- coding: utf-8 -*-
"""
#Analyse motifs
@author: baren
"""
import os
base_dir = "C:\Users\baren\OneDrive\CS_DST Quarter 3\DL\Project\deep-learning-gan\src"

# run Homer:


epochs = ['0'+str(i)+'0' for i in range(0,2)]
epochs.append('100')
epochs.append('109')
for ep in epochs:
    filename = '../../dGAN_dna_50_160/dGAN_generated_sequences_epoch_'+ep+'.txt'
    
    fasta_pos = "../fasta_files_Homer/epoch"+str(ep)+".fasta"
    fasta_neg = "../Data/human_neg_samp.fasta"
    #convert sequences to fasta files
    os.system("py "+ "convertFASTA.py" +' '+filename+ ' ' +fasta_pos+" id_ep_"+str(ep))
    
    #change directory to where we want Homer results to be in
    dir_ep = "../Homer_results/epoch"+str(ep)
    if not os.path.exists(dir_ep):
        os.makedirs(dir_ep)
    
    #run Homer - i think this has to be done in cwgin
    os.system("findmotifs.pl "+fasta_pos+" fasta "+ dir_ep + " -fasta " +fasta_neg )