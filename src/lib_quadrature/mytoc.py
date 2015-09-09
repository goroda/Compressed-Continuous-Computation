import numpy as np

totstring = "#include \"legquadrules.h\" \n#include <string.h> \n "
for ii in xrange(2,200):
    name = "legrules/quad{0}.dat".format(ii)

    dat = np.loadtxt(name,skiprows=1,dtype=str)
     
    #print dat
    w = dat[:,1]
    p = dat[:,0]
    #w = [str(ww) for ww in dat[:,1] ]
    #p = [str(pp) for pp in dat[:,0] ]

    wts = "static const double w{0}[{0}] = ".format(ii) +  "{" + ",".join(w) + "};"
    pts = "static const double pts{0}[{0}] = ".format(ii) +  "{" + ",".join(p) + "};"

    totstring = totstring + "\n" + wts + "\n" + pts + "\n"
    #print wts
    #print pts
    #print "\n"


totstring = totstring + "\n \n \n" 

func = "void getPtsWts(size_t n, double * pts, double * wts)\n {  \n"
func = func + "\t switch (n){\n"
for ii in xrange(2,200):
    func = func + "\t\t case {0}:\n \t\t\t memmove(pts,pts{0},{0}*sizeof(double)); \n \t\t\t memmove(wts,w{0},{0}*sizeof(double)); \n \t\t\t break; \n".format(ii) 

func = func + "\t}"
func = func + " \n}"

#print func

totstring = totstring + func + "\n"

text_file  = open("legquadrules.c","w")
text_file.write("%s" % totstring)
text_file.close()

