// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation

// This file is part of the Compressed Continuous Computation (C3) Library
// Author: Alex A. Gorodetsky 
// Contact: alex@alexgorodetsky.com

// All rights reserved.

// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//Code





#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "stringmanip.h"
#include "CuTest.h"

void Test_strip_blank_ends(CuTest * tc) {

    printf("Testing Function: strip_blank_ends\n");

    char str[]= "           Al Goro                     ";
    strip_blank_ends(str);
    int out = strcmp(str,"Al Goro");
    CuAssertIntEquals(tc,0,out);
}

void Test_strip_ends(CuTest * tc) {

    printf("Testing Function: strip_ends\n");

    char str[]= "<<<<<<Al Goro$$$$$$$$$$$$$$";
    strip_ends(str,'<','$');
    int out = strcmp(str,"Al Goro");
    CuAssertIntEquals(tc,0,out);
}

void Test_bite_string(CuTest * tc) {

    printf("Testing Function: bite_string\n");

    char str[]= "Al Goro: whatup";
    

    char * pre = bite_string(str,':');

    int out;

    out = strcmp(pre,"Al Goro");
    CuAssertIntEquals(tc,0,out);
    out = strcmp(str," whatup");
    CuAssertIntEquals(tc,0,out);


    
    free(pre);
}

void Test_bite_string2(CuTest * tc) {

    printf("Testing Function: bite_string2\n");

    char str[]= "Al Goro: whatup";
    
    char * pre = bite_string2(str,"what");
    
    //printf("pre=\n%sb\n",pre);
    //printf("str=\n%sb\n",str);

    int out;

    out = strcmp(pre,"Al Goro: ");
    CuAssertIntEquals(tc,0,out);
    out = strcmp(str,"up");
    CuAssertIntEquals(tc,0,out);
    
    free(pre);
}


void Test_parse_string(CuTest * tc) {

    printf("Testing Function: parse_string\n");

    char str[]= "Al Goro: whatup: 1.2545 : $";
    
    size_t N;
    char tok = ':';
    char ** vals = parse_string(str,tok,&N);
    
    int out;
    out = strcmp(vals[0],"Al Goro");
    CuAssertIntEquals(tc,0,out);
    out = strcmp(vals[1]," whatup");
    CuAssertIntEquals(tc,0,out);
    out = strcmp(vals[2]," 1.2545 ");
    CuAssertIntEquals(tc,0,out);
    out = strcmp(vals[3]," $");
    CuAssertIntEquals(tc,0,out);

    size_t ii;
    for (ii = 0; ii < N; ii++){
        free(vals[ii]);
    }
    free(vals);
}

void Test_concat_string(CuTest * tc) {

    printf("Testing Function: concat_string\n");
    
    char str1[]= "";
    char * str2 = NULL;
    char str3[] = "whatup ";

    char * s1;
    int out;

    s1 = concat_string(str1,str3);
    out = strcmp(s1,"whatup ");
    CuAssertIntEquals(tc,0,out);
    free(s1);

    s1 = concat_string(str2,str3);
    out = strcmp(s1,"whatup ");
    CuAssertIntEquals(tc,0,out);
    free(s1);

    s1 = concat_string(str1,str3);
    out = strcmp(s1,"whatup ");
    CuAssertIntEquals(tc,0,out);
    free(s1);

    s1 = concat_string(str1,str2);
    out = strcmp(s1,"");
    CuAssertIntEquals(tc,0,out);
    free(s1);
}

void Test_concat_string_ow(CuTest * tc) {

    printf("Testing Function: concat_string_ow\n");
    
    char * str1 = NULL;
    char str2[] = "whatup ";
    char str3[]= "crazy?!";

    int out;

    concat_string_ow(&str1,str2);
    out = strcmp(str1,"whatup ");
    CuAssertIntEquals(tc,0,out);
    concat_string_ow(&str1,str3);
    out = strcmp(str1,"whatup crazy?!");
    CuAssertIntEquals(tc,0,out);
    free(str1);
}

void Test_serialize_char(CuTest * tc){

    printf("Char serialization tests and experiments \n");
    char a = 'a';
    unsigned char buffer[32];

    serialize_char(buffer, a);
    printf("Size of char = %zu\n",sizeof(char));
    printf("Size of a = %zu\n",sizeof(a));
    printf("Size of 'a' = %zu\n",sizeof('a'));
    printf("Serialized value is {%c}\n",buffer[0]);
    CuAssertIntEquals(tc,0,0);
}

void Test_serialize_int(CuTest * tc){

    printf("Int serialization tests and experiments \n");
    printf("maximum int = %d\n",INT_MAX);
    printf("size of int in bytes= %zu\n",sizeof(int));
    int a = 18505;
    int value;
    unsigned char buffer[sizeof(int)];

    serialize_int(buffer,a);
    printf(" serialized is %s\n",buffer);
    deserialize_int(buffer,&value);
    printf(" deserialized is %d\n",value);
    CuAssertIntEquals(tc,value,a);
    a = INT_MAX;
    serialize_int(buffer,a);
    deserialize_int(buffer,&value);
    CuAssertIntEquals(tc,value,a);
    a = INT_MIN;
    serialize_int(buffer,a);
    deserialize_int(buffer,&value);
    CuAssertIntEquals(tc,value,a);
}

void Test_serialize_sizet(CuTest * tc){

    printf("sizet serialization tests and experiments \n");
    printf("size of sizet in bytes= %zu\n",sizeof(size_t));
    printf("size of long in bytes= %zu\n",sizeof(long));
    printf("size of long long in bytes= %zu\n",sizeof(long long ));
    printf("size of unsigned long long in bytes= %zu\n",
                                sizeof(unsigned long long ));
    
    size_t a = 20423;
    size_t value;
    unsigned char buffer[sizeof(size_t)];
    
    serialize_size_t(buffer,a);
    deserialize_size_t(buffer,&value);
    CuAssertIntEquals(tc,value,a);
    a = ULLONG_MAX;
    serialize_size_t(buffer,a);
    deserialize_size_t(buffer,&value);
    CuAssertIntEquals(tc,value,a);
    a = 0;
    serialize_size_t(buffer,a);
    deserialize_size_t(buffer,&value);
    CuAssertIntEquals(tc,value,a);
}

void Test_serialize_double(CuTest * tc){

    printf("doubleserialization tests and experiments \n");
    printf("size of double in bytes= %zu\n",sizeof(double));
    printf("DBL_MAX = %G \n",DBL_MAX);
    printf("DBL_MIN = %G \n",DBL_MIN);
    printf("DBL_EPSILON = %G\n",DBL_EPSILON);

    
    double a = 20423.231;
    double value;
    unsigned char buffer[sizeof(double)];
    
    serialize_double(buffer,a);
    deserialize_double(buffer,&value);
    CuAssertDblEquals(tc,value,a,0.0);
    a = DBL_MAX;
    serialize_double(buffer,a);
    deserialize_double(buffer,&value);
    CuAssertDblEquals(tc,value,a,0.0);
    a = 0.0;
    serialize_double(buffer,a);
    deserialize_double(buffer,&value);
    CuAssertDblEquals(tc,value,a,0.0);
}

void Test_serialize_doublep(CuTest * tc){

    printf("double pointer serialization tests and experiments \n");
    
    size_t N = 4;
    double a[4] = {-0.2314, 0.0, 200.0222, 34.223112};

    double * value = NULL;
    size_t Nval;
    unsigned char buffer[4*sizeof(double) + sizeof(size_t)];
    
    serialize_doublep(buffer,a,N);
    deserialize_doublep(buffer,&value,&Nval);
    
    CuAssertIntEquals(tc,4,Nval);
    size_t ii;
    for (ii = 0; ii < 4; ii++){
        CuAssertDblEquals(tc,a[ii],value[ii],0.0);
    }
    free(value);
}


void Test_readtxt_double_array(CuTest * tc)
{
    printf("Testing function: readtxt_double_array\n");

    char line[] = "1e-0 0.21 2.134 3000\n 021.32 0.1124 0.3128 0.987";

    size_t nrows,ncols;
    double * data = NULL;

    data = readtxt_double_array(line,&nrows,&ncols);
    CuAssertIntEquals(tc,2,nrows);
    CuAssertIntEquals(tc,4,ncols);

    CuAssertDblEquals(tc,1,data[0],1e-15);
    CuAssertDblEquals(tc,0.21,data[1],1e-15);
    CuAssertDblEquals(tc,2.134,data[2],1e-15);
    CuAssertDblEquals(tc,3000.0,data[3],1e-15);

    CuAssertDblEquals(tc,21.32,data[4],1e-15);
    CuAssertDblEquals(tc,0.1124,data[5],1e-15);
    CuAssertDblEquals(tc,0.3128,data[6],1e-15);
    CuAssertDblEquals(tc,0.987,data[7],1e-15);

    free(data); data = NULL;

}

void Test_readfile_double_array(CuTest * tc)
{
    printf("Testing function: readfile_double_array\n");

    char fileout[] = "readarr.txt";
    FILE * fp = fopen(fileout,"wt");
    fprintf(fp, "1e-0 0.21 2.134 3000\n");
    fprintf(fp, "021.32 0.1124 0.3128 0.987\n");
    fclose(fp);

    fp = fopen(fileout,"rt");
    
    size_t nrows,ncols;
    double * data = NULL;

    data = readfile_double_array(fp,&nrows,&ncols);

    CuAssertIntEquals(tc,2,nrows);
    CuAssertIntEquals(tc,4,ncols);

    CuAssertDblEquals(tc,1,data[0],1e-15);
    CuAssertDblEquals(tc,0.21,data[1],1e-15);
    CuAssertDblEquals(tc,2.134,data[2],1e-15);
    CuAssertDblEquals(tc,3000.0,data[3],1e-15);

    CuAssertDblEquals(tc,21.32,data[4],1e-15);
    CuAssertDblEquals(tc,0.1124,data[5],1e-15);
    CuAssertDblEquals(tc,0.3128,data[6],1e-15);
    CuAssertDblEquals(tc,0.987,data[7],1e-15);

    free(data); data = NULL;

    fclose(fp);
}

CuSuite * StringUtilGetSuite(){
    //printf("Generating Suite: VectorUtil\n");
    //printf("----------------------------\n");

    CuSuite * suite = CuSuiteNew();
    SUITE_ADD_TEST(suite, Test_strip_blank_ends);
    SUITE_ADD_TEST(suite, Test_strip_ends);
    SUITE_ADD_TEST(suite, Test_bite_string);
    SUITE_ADD_TEST(suite, Test_bite_string2);
    SUITE_ADD_TEST(suite, Test_parse_string);
    SUITE_ADD_TEST(suite, Test_concat_string);
    SUITE_ADD_TEST(suite, Test_concat_string_ow);
    SUITE_ADD_TEST(suite, Test_serialize_char);
    SUITE_ADD_TEST(suite, Test_serialize_int);
    SUITE_ADD_TEST(suite, Test_serialize_sizet);
    SUITE_ADD_TEST(suite, Test_serialize_double);
    SUITE_ADD_TEST(suite, Test_serialize_doublep);
    SUITE_ADD_TEST(suite, Test_readtxt_double_array);
    SUITE_ADD_TEST(suite, Test_readfile_double_array);

    return suite;
}

void RunAllTests(void) {
    
    printf("Running Test Suite: lib_stringmanip\n");

    CuString * output = CuStringNew();
    CuSuite * suite = CuSuiteNew();
    
    CuSuite * st = StringUtilGetSuite();
    CuSuiteAddSuite(suite, st);
    CuSuiteRun(suite);
    CuSuiteSummary(suite, output);
    CuSuiteDetails(suite, output);
    printf("%s \n", output->buffer);
    
    CuSuiteDelete(st);
    CuStringDelete(output);
    free(suite);
}

int main(void) {
    RunAllTests();
}
