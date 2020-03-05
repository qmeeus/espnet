import sys
import os
import pickle

# This script will remove all suffixes "_A0"/"_A1"/... etc from the files in the .flist and stores them, so that after processing the flist the suffix _A0 can be attached in the wav.scp file (otherwise the process_flist.pl would have to be adjusted completely)
# The required inputs are the location of the folder containing 'temp.flist' and the name to call the new flist, e.g. 'train_s' (will become train_s.flist)

loc = sys.argv[1]
name = sys.argv[2]

wavdic = {}
suffix_dic = {}

with open(os.path.join(loc,'temp.flist'),'r') as fid:
	with open(os.path.join(loc,name+'.flist'),'w+') as pid:
		line = fid.readline()
		while line: 
			sep = line.split("_")	
			if len(sep) > 1:  # contains suffix _XX.wav
				if (sep[0]) not in wavdic:  # check if file without suffix is not there
					wavdic[sep[0]] = 1  # remove _A0.wav
					newline = sep[0]+'.wav\n'  
					pid.write(newline)
				if (sep[0]) not in suffix_dic:
					suffix_dic[sep[0]] = sep[1]
				else:
					suffixes = [suffix_dic[sep[0]],sep[1]]
					suffix_dic[sep[0]] = (sorted(suffixes))[0]  # choose _A0 or _B0
			else:
				sep = line.split(".")  
				if (sep[0]) not in wavdic:  # check if file with suffix has been handled
					wavdic[sep[0]] = 1
					pid.write(line)

			line = fid.readline()


pickle_out = open("suffixes.pickle","wb")  # save the suffices
pickle.dump(suffix_dic, pickle_out)
pickle_out.close()
