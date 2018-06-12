# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:13:00 2018

@author: baren
"""

import sys

#File input
fileInput = open(sys.argv[1], "r")

#File output
fileOutput = open(sys.argv[2], "w")

identifier = sys.argv[3]
#Seq count
count = 1 ;

#Loop through each line in the input file
print("Converting to FASTA...")
for strLine in fileInput:

    #Strip the endline character from each input line
    strLine = strLine.rstrip("\n")

    #Output the header
    fileOutput.write(">"+identifier + str(count) + "\n")
    fileOutput.write(strLine + "\n")

    count = count + 1
print ("Done.")

#Close the input and output file
fileInput.close()
fileOutput.close()