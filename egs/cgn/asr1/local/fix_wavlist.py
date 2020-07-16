import sys
import os
import pickle

# This script makes changes to the wav.scp file so that if necessary the utterances point to wavfiles with a suffix _A0.
# This suffix has been removed before in fix_flist.py so that the process_flist script does not crash.
# Required inputs are the location of the suffixes.pickle file created by fix_flist.py and the name of the wav.scp file
# (e.g. train_s will become train_s_wav.scp). The wav.scp file is expected to be called 'temp.scp'

loc = sys.argv[1]
name = sys.argv[2]

pickle_in = open(os.path.join(loc,'suffixes.pickle'),"rb")
suffix_dic = pickle.load(pickle_in)

with open(os.path.join(loc,'temp.scp'),'r') as fid:
	with open(os.path.join(loc,name+'_wav.scp'),'w+') as pid:
		line = fid.readline()
		while line:
			splitline = (line.split(" "))
			wavloc = splitline[4]
			wavname = (wavloc.split('.'))[0]
			if wavname in suffix_dic:  # remove .wav 
				splitline[4] = wavname+'_'+(suffix_dic[wavname])[:2]+'.wav'  # add _A0.wav for example
			newline = " ".join(splitline)
			pid.write(newline)
			line = fid.readline()
