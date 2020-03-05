#!/usr/bin/python

import sys
import os

comp = sys.argv[1]
bestpath = sys.argv[2]
decodedir = sys.argv[3]
wavdir = sys.argv[4]

wavlist=os.listdir(os.path.join(wavdir,'comp-'+str(comp),'vl'))  # list all wavs for this component
for i in range(len(wavlist)):
	wavlist[i] = (wavlist[i])[0:8]  # filter out suffixes like _A1.wav

wavcount = {}

## Create filtered versions of <bestpath>.tra and test_filt.txt based on if the utterance belonged to this component. This way we can compute WERs for every component separately

with open(os.path.join(decodedir,'compscoring',str(comp)+'.tra'),'w+') as tp:
	with open(os.path.join(decodedir,'scoring',bestpath+'.tra'),'r') as tr:
		line = tr.readline()
		while line:
			uttid = (line.split(' '))[0]
			wav = (uttid.split('-'))[1]
			wavname = wav[0:8]  # go from Vxxxxx-fvxxxxxx(_xx).xx to fvxxxxxx as in wavlist

			fullwav = uttid.split('.')[0]
			if fullwav not in wavcount:
				wavcount[fullwav] = 1			

			if wavname in wavlist:			
				tp.write(line)
			line = tr.readline()

with open(os.path.join(decodedir,'compscoring','test_filt_'+str(comp)+'.txt'),'w+') as fp:
	with open(os.path.join(decodedir,'scoring','test_filt.txt'),'r') as fr:
		line = fr.readline()
		while line:
			uttid = (line.split(' '))[0]
			wav = (uttid.split('-'))[1]
			wavname = wav[0:8]  # go from Vxxxxx-fvxxxxxx(_xx).xx to fvxxxxxx as in wavlist
			if wavname in wavlist:
				fp.write(line)
			line = fr.readline()
	
