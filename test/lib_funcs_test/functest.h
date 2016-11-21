#ifndef FUNCTEST_H
#define FUNCTEST_H

#include "CuTest.h"

CuSuite * ChebGetSuite();
CuSuite * LegGetSuite();
CuSuite * HermGetSuite();
CuSuite * LelmGetSuite();
CuSuite * StandardPolyGetSuite();
CuSuite * PolyAlgorithmsGetSuite();
CuSuite * PolySerializationGetSuite();
CuSuite * LinkedListGetSuite();
CuSuite * PiecewisePolyGetSuite();
CuSuite * PolyApproxSuite();
CuSuite * PolyRegressionSuite();


#endif
