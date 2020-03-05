#!/usr/bin/python

import sys
import os

dir = sys.argv[1]
suffix = sys.argv[2]

wavdir = '/users/spraak/spchdata/cgn/wav'

comps = ['o','k','l','j','m','n','g','f','b','h','a','i','c','d']
wavdict = {}
durdict = {}
durdict_seg = {}

for comp in comps:
	durdict[comp] = 0
	durdict_seg[comp] = 0
	wavlist=os.listdir(os.path.join(wavdir,'comp-'+str(comp),'vl'))
	for i in range(len(wavlist)):
		wavname = (wavlist[i])[0:8]
		wavdict[wavname] = comp

with open(os.path.join(dir,'reco2dur_'+suffix)) as fp:
	line = fp.readline()
	while line:
		wav = line.split(' ')[0]
		dur = line.split(' ')[1]
		comp = wavdict[wav]
		durdict[comp] += float(dur)
		line = fp.readline()

with open(os.path.join(dir,'reco2dur_seg_'+suffix)) as fp:
	line = fp.readline()
	while line:
		wav = line.split(' ')[0]
		dur = line.split(' ')[1]
		comp = wavdict[wav]
		durdict_seg[comp] += float(dur)
		line = fp.readline()

for comp in comps:
	durdict[comp] = round(durdict[comp]/3600,2)
	durdict_seg[comp] = round(durdict_seg[comp]/3600,2)
print 'Duration of all wav-files (in hours) per component'
print durdict

print 'Duration of all utterances (in hours) per component'
print durdict_seg
