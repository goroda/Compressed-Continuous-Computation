import numpy as np

totstring = "#include <string.h> \n "

name = "legpolytens.dat"
dat = np.loadtxt(name,skiprows=1,dtype=str)
    
print dat
w = dat

w  = "static const double lpolycoeffs[8000000] = {" + ",".join(w) + "};"

totstring = totstring + "\n" + w
    #print wts
    #print pts
    #print "\n"


totstring = totstring + "\n \n \n" 

text_file  = open("legtens.h","w")
text_file.write("%s" % totstring)
text_file.close()

