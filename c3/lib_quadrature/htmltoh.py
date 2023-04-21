from lxml import html
import numpy as np
import bs4

with open ("leggaussquad.html","r") as myfile:
    data = myfile.read().replace('\n', '')

soup = bs4.BeautifulSoup(data,"lxml")


#for ii in xrange(2,65):

totstring = "#include \"legquadrules.h\" \n#include <string.h> \n "
for ii in xrange(2,65):
    name = "n{0}".format(ii)
    #print name
    temp = soup.find(id=name)
    tbl = temp.findAll("table")[1]
    dat = np.array([map(str,[a.string for a in row.findAll("td")]) for row in tbl.findAll("tr")][1:])[:,1:]
    #wts = [float(d) for d in dat[:,0]]
    #pts = [float(d) for d in dat[:,1]]

    wts = "static const double w{0}[{0}] = ".format(ii) +  "{" + ",".join(dat[:,0]) + "};"
    pts = "static const double pts{0}[{0}] = ".format(ii) +  "{" + ",".join(dat[:,1]) + "};"

    totstring = totstring + "\n" + wts + "\n" + pts + "\n"
    #print wts
    #print pts
    #print "\n"


totstring = totstring + "\n \n \n" 

func = "void getPtsWts(size_t n, double * pts, double * wts)\n {  \n"
func = func + "\t switch (n){\n"
for ii in xrange(2,65):
    func = func + "\t\t case {0}:\n \t\t\t memmove(pts,pts{0},{0}*sizeof(double)); \n \t\t\t memmove(wts,w{0},{0}*sizeof(double)); \n \t\t\t break; \n".format(ii) 

func = func + "\t}"
func = func + " \n}"

#print func

totstring = totstring + func + "\n"

text_file  = open("legquadrules.c","w")
text_file.write("%s" % totstring)
text_file.close()

