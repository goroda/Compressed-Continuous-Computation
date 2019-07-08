import numpy as np
v1 = 0
v2 = 1
if v1:
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
elif v2 == 1:
    totstring = "#include \"legquadrules.h\" \n#include <string.h> \n "
    wts = "static double w1[1] = ".format(1) +  "{ 1.0 };"
    pts = "static double pts1[1] = ".format(1) +  "{ 0.0 };"
    totstring = totstring + "\n" + wts + "\n" + pts + "\n"
    for ii in xrange(2,200):
        name = "legrules/quad{0}.dat".format(ii)
        dat = np.loadtxt(name,skiprows=1,dtype=str)
    
        #print dat
        w = dat[:,1]
        p = dat[:,0]
        #w = [str(ww) for ww in dat[:,1] ]
        #p = [str(pp) for pp in dat[:,0] ]

        wts = "static double w{0}[{0}] = ".format(ii) +  "{" + ",".join(w) + "};"
        pts = "static double pts{0}[{0}] = ".format(ii) +  "{" + ",".join(p) + "};"

        totstring = totstring + "\n" + wts + "\n" + pts + "\n"
        #print wts
        #print pts
        #print "\n"
    totstring = totstring + "\n \n \n"
        
    totstring = totstring + "\n" + "static double * legwts[199] = { \n"
    for ii in xrange(1,199):
        totstring = totstring + "w{0}, ".format(ii)
    totstring = totstring + "w199 ".format(ii)
    totstring = totstring + "}; \n \n"

    totstring = totstring + "\n" + "static double * legpts[199] = { \n"
    for ii in xrange(1,199):
        totstring = totstring + "pts{0}, ".format(ii)
    totstring = totstring + "pts199 "
    totstring = totstring + "}; \n \n"


    func = "int getLegPtsWts2(size_t n, double * pts, double * wts)\n {  \n"
    func = func + "\t if ((n < 1) || (n > 200)){\n "
    func = func + "\t \t return 1;\n"
    func = func + "\t }\n"
    func = func + "\t pts = legpts[n-1];\n"
    func = func + "\t wts = legwts[n-1];\n"
    func = func + "\t return 0;"
    func = func + " \n}\n\n"

    totstring = totstring + func
    
    text_file  = open("legquadrules2.c","w")
    text_file.write("%s" % totstring)
    text_file.close()
