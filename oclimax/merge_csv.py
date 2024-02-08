#! /usr/bin/env python

from __future__ import print_function

import sys
import csv
import numpy as np
import math

al = 2
#print(sys.argv)
nf=len(sys.argv)-al

f = open(sys.argv[al],'r')
full_text = f.read()
f.close()
header = full_text.split('\n')[0:4]

for i in range(al,len(sys.argv)):
    csvfile=sys.argv[i]
    print(csvfile)
    sim = list(csv.reader(open(csvfile,'r')))
    ii=0
    while ii<len(sim):
        if not sim[ii]:
            del sim[ii]
            continue
        try:
            float(sim[ii][0])
        except ValueError:
            del sim[ii]
            continue
        ii=ii+1

    for row in range(len(sim)):
        for col in range(len(sim[0])):
            try:
                float(sim[row][col])
            except ValueError:
                sim[row][col]='0.0'
                continue

    if i==al:
        tsim=[[float(sim[row][col]) for row in range(len(sim))] for col in range(len(sim[0]))]
    else:
        ttsim=[[float(sim[row][col]) for row in range(len(sim))] for col in range(len(sim[0]))]
        for j in range(len(tsim[2])):
            tsim[2][j]=tsim[2][j]+ttsim[2][j]

x=[tsim[0][i] for i in range(len(tsim[0]))]
y=[tsim[1][i] for i in range(len(tsim[1]))]
z=[tsim[2][i] for i in range(len(tsim[2]))]
xi=np.sort(np.asarray(list(set(x))))
yi=np.sort(np.asarray(list(set(y))))
zi = np.zeros((len(yi),len(xi)))

for i in range(len(yi)):
    for j in range(len(xi)):
        k = i+j*len(yi)
        if z[k]>=0.0:
            zi[i][j]=z[k]
        elif z[k]==-1:
            zi[i][j]=np.nan
        else:
            zi[i][j]=0.0

f = open(sys.argv[1]+'.csv','w')
for line in header:
    print(line, file=f)
for i in range(len(xi)):
    for j in range(len(yi)):
        print(str(xi[i]),',',str(yi[j]),',',str(zi[j][i]), file=f)
    print(" ", file=f)
f.close()
