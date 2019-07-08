// Copyright (c) 2015-2016, Massachusetts Institute of Technology
// Copyright (c) 2016-2017 Sandia Corporation
// Copyright (c) 2017 NTESS, LLC.

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


/** \file polynomials.c
 * Provides routines for manipulating orthogonal polynomials
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <assert.h>
#include "futil.h"

//#define ZEROTHRESH 1e-20
/* #define ZEROTHRESH  1e0 * DBL_EPSILON */
/* #define ZEROTHRESH  1e-3 * DBL_EPSILON */
#define ZEROTHRESH  1e0 * DBL_EPSILON
/* #define ZEROTHRESH  1e0 * DBL_MIN */
/* #define ZEROTHRESH  1e-200 */
//#define ZEROTHRESH 0.0
/* #define ZEROTHRESH  1e0 * DBL_EPSILON */
//#define ZEROTHRESH  1e-12

#include "stringmanip.h"
#include "array.h"
#include "polynomials.h"
#include "hpoly.h"
#include "lib_quadrature.h"
#include "linalg.h"
#include "legtens.h"
#include "fourier.h"

enum SPACE_MAP {SM_LINEAR,SM_ROSENBLATT};

struct SpaceMapping
{
    enum SPACE_MAP map;
    int set;
    double lin_slope;
    double lin_offset;
    double inv_lin_slope;
    double inv_lin_offset;
};


static struct SpaceMapping * space_mapping_create(enum SPACE_MAP map_type)
{
    struct SpaceMapping * map = malloc(sizeof(struct SpaceMapping));
    if (map == NULL){
        fprintf(stderr,"Failure to allocate space mapping for polynomials\n");
        exit(1);
    }

    map->map = map_type;
    map->set = 0;
    if (map_type == SM_LINEAR){
        map->lin_slope = 1;
        map->lin_offset = 0;
        map->inv_lin_slope = 1;
        map->inv_lin_offset = 0;
    }
    else if (map_type == SM_ROSENBLATT){
        fprintf(stderr,"Rosenblatt mapping not yet implemented\n");
        exit(1);
    }
    else{
        fprintf(stderr,"Mapping type %d is not defined\n",map_type);
        exit(1);
    }
    return map;
}

static struct SpaceMapping * space_mapping_copy(const struct SpaceMapping * map)
{
    struct SpaceMapping * map_out = space_mapping_create(map->map);
    map_out->set            = map->set;
    map_out->lin_slope      = map->lin_slope;
    map_out->lin_offset     = map->lin_offset;
    map_out->inv_lin_slope  = map->inv_lin_slope;
    map_out->inv_lin_offset = map->inv_lin_offset;
    return map_out;
}

static void space_mapping_free(struct SpaceMapping * map)
{
    if (map != NULL){
        free(map); map = NULL;
    }
}

/********************************************************//**
*   Serialize the mapping
*************************************************************/
static unsigned char *
serialize_space_mapping(unsigned char * ser, 
                        struct SpaceMapping * p,
                        size_t * totSizeIn)
{
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff
    
    size_t totsize =
        sizeof(enum SPACE_MAP) +
        sizeof(int) +
        4*sizeof(double); 

    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }
    
    unsigned char * ptr = serialize_int(ser, p->map);
    ptr = serialize_int(ptr,p->set);
    ptr = serialize_double(ptr, p->lin_slope);
    ptr = serialize_double(ptr, p->lin_offset);
    ptr = serialize_double(ptr, p->inv_lin_slope);
    ptr = serialize_double(ptr, p->inv_lin_offset);
    return ptr;
}

/********************************************************//**
*   Deserialize space mapping
*************************************************************/
static unsigned char * 
deserialize_space_mapping(
    unsigned char * ser, 
    struct SpaceMapping ** smap)
{
    enum SPACE_MAP map;
    int map_int;

    unsigned char * ptr = deserialize_int(ser,&map_int);
    map = (enum SPACE_MAP) map_int;

    *smap = space_mapping_create(map);
    ptr = deserialize_int(ptr, &((*smap)->set));
    ptr = deserialize_double(ptr, &((*smap)->lin_slope));
    ptr = deserialize_double(ptr, &((*smap)->lin_offset));
    ptr = deserialize_double(ptr, &((*smap)->inv_lin_slope));
    ptr = deserialize_double(ptr, &((*smap)->inv_lin_offset));
    
    return ptr;
}


/********************************************************//**
    Save a mapping in text format
************************************************************/
static void space_mapping_savetxt(const struct SpaceMapping * f,
                                  FILE * stream, size_t prec)
{
    assert (f != NULL);
    fprintf(stream,"%d ",f->map);
    fprintf(stream,"%d ",f->set);
    fprintf(stream,"%3.*G ",(int)prec,f->lin_slope);
    fprintf(stream,"%3.*G ",(int)prec,f->lin_offset);
    fprintf(stream,"%3.*G ",(int)prec,f->inv_lin_slope);
    fprintf(stream,"%3.*G ",(int)prec,f->inv_lin_offset);
}

/********************************************************//**
    Load a mapping in text format
************************************************************/
static struct SpaceMapping *
space_mapping_loadtxt(FILE * stream)//l, size_t prec)
{

    enum SPACE_MAP maptype;
    int maptypeint;
    int num = fscanf(stream,"%d ",&maptypeint);
    maptype = (enum SPACE_MAP)maptypeint;
    assert (num == 1);

    struct SpaceMapping * map = space_mapping_create(maptype);
    num = fscanf(stream,"%d ",&(map->set));
    assert (num == 1);
    
    num = fscanf(stream,"%lG ",&(map->lin_slope));
    assert (num == 1);
    num = fscanf(stream,"%lG ",&(map->lin_offset));
    assert (num == 1);
    num = fscanf(stream,"%lG ",&(map->inv_lin_slope));
    assert (num == 1);
    num = fscanf(stream,"%lG ",&(map->inv_lin_offset));
    assert (num == 1);

    return map;
}


// original space to normalized space
double space_mapping_map(struct SpaceMapping * map, double x)
{
    if (map->map == SM_LINEAR){
        return map->lin_slope * x + map->lin_offset;
    }
    else{
        fprintf(stderr, "NONLIENAR MAPPINGS NOT YET IMPLEMENTED\n");
        exit(1);
    }
}

double space_mapping_map_deriv(struct SpaceMapping * map, double x)
{
    (void)(x);
    if (map->map == SM_LINEAR){
        return map->lin_slope;
    }
    else{
        fprintf(stderr, "NONLIENAR MAPPINGS NOT YET IMPLEMENTED\n");
        exit(1);
    }
}

// normalized space to original space
double space_mapping_map_inverse(struct SpaceMapping * map, double x)
{
    if (map->map == SM_LINEAR){
        return map->inv_lin_slope * x + map->inv_lin_offset;
    }
    else{
        fprintf(stderr, "NONLINEAR MAPPINGS NOT YET IMPLEMENTED\n");
        exit(1);
    }
}

double space_mapping_map_inverse_deriv(struct SpaceMapping * map, double x)
{
    (void)(x);
    if (map->map == SM_LINEAR){
        return map->inv_lin_slope;
    }
    else{
        fprintf(stderr, "NONLIENAR MAPPINGS NOT YET IMPLEMENTED\n");
        exit(1);
    }
}


// Recurrence relationship sequences
inline static double zero_seq(size_t n){ (void)n; return (0.0); }
/* inline static double one_seq(size_t n) { (void)n; return (1.0); } */
inline static double none_seq(size_t n){ (void)n; return (-1.0); }
inline static double two_seq(size_t n) { (void)n; return (2.0); }
/* inline static double n_seq(size_t n) { return ((double) n); } */
/* inline static double nn_seq (size_t n) { return -n_seq(n); } */
/* inline static double lega_seq (size_t n) { return ( (double)(2.0 * n -1.0) / (double) n);} */
/* inline static double legc_seq (size_t n) { return ( -((double)n - 1.0)/ (double) n );} */


static const double legaseqnorm[201] = {
0.0000000000000000000000000,
1.7320508075688772935737253,
1.9364916731037084426068906,
1.9720265943665386808867149,
1.9843134832984429429111536,
1.9899748742132399093648226,
1.9930434571835663367907546,
1.9948914348241344529156019,
1.9960899278339139998987573,
1.9969111950679364953821493,
1.9974984355438178915695749,
1.9979328159850827787737126,
1.9982631347136331423841246,
1.9985201625794738002940554,
1.9987240828047460811708186,
1.9988885800753266486651932,
1.9990231989649344737647318,
1.9991347609372268759927310,
1.9992282461607312885921994,
1.9993073592865872621432075,
1.9993749023132204957258970,
1.9994330262111444065009289,
1.9994834043566154743211405,
1.9995273543594641291838362,
1.9995659251169712031020662,
1.9995999599919979994094507,
1.9996301433163012802865163,
1.9996570350656427370343235,
1.9996810970242025987542861,
1.9997027127445487499745463,
1.9997222029294191164312117,
1.9997398373972733498548163,
1.9997558444720195391930417,
1.9997704184116869836722319,
1.9997837253305382904924534,
1.9997959079539561241378037,
1.9998070894618131628688376,
1.9998173766146948726477386,
1.9998268623119294251254049,
1.9998356276964555233591550,
1.9998437438960074911519643,
1.9998512734706997544817733,
1.9998582716222576653730680,
1.9998647872087154521263119,
1.9998708635995425604743592,
1.9998765393992465586080545,
1.9998818490620738438532863,
1.9998868234161445738743604,
1.9998914901119566046557366,
1.9998958740074785727074830,
1.9998999974998749922178512,
1.9999038808121516333851653,
1.9999075422415889522788357,
1.9999109983756762928906042,
1.9999142642803164142926986,
1.9999173536642966251576647,
1.9999202790233863698352718,
1.9999230517668953255793970,
1.9999256823290901509881587,
1.9999281802675053497939453,
1.9999305543498809755115567,
1.9999328126312063734840163,
1.9999349625221361967906952,
1.9999370108498654878071213,
1.9999389639123990025908889,
1.9999408275270214743832980,
1.9999426070736663534153227,
1.9999443075337875691619566,
1.9999459335252594405609242,
1.9999474893337618725376939,
1.9999489789410496206473189,
1.9999504060504542197682087,
1.9999517741099239061773607,
1.9999530863328694717815162,
1.9999543457170516339753766,
1.9999555550617174208343982,
1.9999567169831686592367317,
1.9999578339289244014115657,
1.9999589081906205670141272,
1.9999599419157738604399185,
1.9999609371185228226443925,
1.9999618956894464111197912,
1.9999628194045495514557728,
1.9999637099334954661120772,
1.9999645688471560844769875,
1.9999653976245443358231504,
1.9999661976591854849498106,
1.9999669702649787970689249,
1.9999677166815955984928432,
1.9999684380794551698127051,
1.9999691355643157882884461,
1.9999698101815145638701296,
1.9999704629198864438586677,
1.9999710947153898319165674,
1.9999717064544636596331098,
1.9999722989771384023096126,
1.9999728730799214406138944,
1.9999734295184752761939673,
1.9999739690101054293697169,
1.9999744922360733178625866,
1.9999749998437480468264568,
1.9999754924486098081387250,
1.9999759706361164681223674,
1.9999764349634439192238133,
1.9999768859611098577311328,
1.9999773241344898295265148,
1.9999777499652336322380833,
1.9999781639125894913602138,
1.9999785664146428101331163,
1.9999789578894757322230849,
1.9999793387362532517655470,
1.9999797093362411396682068,
1.9999800700537605352685028,
1.9999804212370836657046771,
1.9999807632192748093908491,
1.9999810963189802913647988,
1.9999814208411710151919866,
1.9999817370778407593125137,
1.9999820453086632256753732,
1.9999823458016106016892424,
1.9999826388135361899785550,
1.9999829245907234727581969,
1.9999832033694038013488942,
1.9999834753762447451218287,
1.9999837408288109826979759,
1.9999839999359994880149746,
1.9999842528984506348542841,
1.9999844999089367322906569,
1.9999847411527293949973227,
1.9999849768079470566052233,
1.9999852070458838431332368,
1.9999854320313209394806633,
1.9999856519228215081390720,
1.9999858668730101443622438,
1.9999860770288377853545064,
1.9999862825318329381286964,
1.9999864835183400217297200,
1.9999866801197455758263427,
1.9999868724626930385594725,
1.9999870606692867456861268,
1.9999872448572857672795949,
1.9999874251402881543361253,
1.9999876016279061367387007,
1.9999877744259327734793058,
1.9999879436365005309960211,
1.9999881093582322320878486,
1.9999882716863847930419476,
1.9999884307129861387439598,
1.9999885865269656637496429,
1.9999887392142785871298680,
1.9999888888580245199528412,
1.9999890355385605579790348,
1.9999891793336091811361319,
1.9999893203183612335350344,
1.9999894585655742371881408,
1.9999895941456662814238177,
1.9999897271268057108006139,
1.9999898575749968286869099,
1.9999899855541618157823633,
1.9999901111262190523407481,
1.9999902343511580255896315,
1.9999903552871109908319058,
1.9999904739904215434384907,
1.9999905905157102581825787,
1.9999907049159375333922603,
1.9999908172424637773983633,
1.9999909275451070651049429,
1.9999910358721983849204437,
1.9999911422706345906497027,
1.9999912467859291663333302,
1.9999913494622609072505146,
1.9999914503425206137960865,
1.9999915494683558901721845,
1.9999916468802141375580433,
1.9999917426173838194951998,
1.9999918367180340843811473,
1.9999919292192528146772171,
1.9999920201570831760143343,
1.9999921095665587343929645,
1.9999921974817372009999503,
1.9999922839357328720796131,
1.9999923689607478141661012,
1.9999924525881018556091456,
1.9999925348482614335075824,
1.9999926157708673449481598,
1.9999926953847614525313497,
1.9999927737180123875522508,
1.9999928507979402920362658,
1.9999929266511406438650011,
1.9999930013035072003373810,
1.9999930747802541001730339,
1.9999931471059371559419182,
1.9999932183044743736746382,
1.9999932883991657291437513,
1.9999933574127122330168702,
1.9999934253672343136129172,
1.9999934922842895436076449,
1.9999935581848897396366194,
1.9999936230895174589733757,
1.9999936870181419158341504,
1.9999937499902343445226660,
};

static const double legcseqnorm[201] = {
0.0000000000000000000000000,
0.0000000000000000000000000,
-1.1180339887498948482072100,
-1.0183501544346311125984611,
-1.0062305898749053633539630,
-1.0028530728448139498158037,
-1.0015420209622192481389857,
-1.0009272139219581055366928,
-1.0006007810695147948465422,
-1.0004114379931337589987872,
-1.0002940744071803442008517,
-1.0002174622185106379413169,
-1.0001653302482984141553307,
-1.0001286256356210525864103,
-1.0001020356106936058139534,
-1.0000823011400101476917751,
-1.0000673468701305763445264,
-1.0000558082429209263942635,
-1.0000467628422711159142180,
-1.0000395718327849260524676,
-1.0000337832131310390878717,
-1.0000290710350803458344310,
-1.0000251962155324265491343,
-1.0000219806789858287190270,
-1.0000192899374059475950394,
-1.0000170211317362819157528,
-1.0000150946813898723700489,
-1.0000134483616539496234413,
-1.0000120330427357868583946,
-1.0000108095837772925361039,
-1.0000097465411964244106496,
-1.0000088184587981272660392,
-1.0000080045786190982050334,
-1.0000072878595193167500232,
-1.0000066542232692270346719,
-1.0000060919704783676505502,
-1.0000055913245009481336162,
-1.0000051440726137630963036,
-1.0000047432817343602117627,
-1.0000043830717004995434563,
-1.0000040584333230009967883,
-1.0000037650815046152063745,
-1.0000034993360010045953049,
-1.0000032580241061291089627,
-1.0000030384008288828010663,
-1.0000028380831019170736640,
-1.0000026549953073004908810,
-1.0000024873239751952141291,
-1.0000023334799536602361322,
-1.0000021920666914396246330,
-1.0000020618535444830643141,
-1.0000019417532284092108713,
-1.0000018308027062940946880,
-1.0000017281469339661820195,
-1.0000016330249909989784721,
-1.0000015447582105839689076,
-1.0000014627399899225625515,
-1.0000013864270181358339523,
-1.0000013153317036379693836,
-1.0000012490156195652532892,
-1.0000011870838158338720483,
-1.0000011291798710141289983,
-1.0000010749815775035198584,
-1.0000010241971702458106327,
-1.0000009765620231633317591,
-1.0000009318357490399825985,
-1.0000008897996482624395340,
-1.0000008502544599036081924,
-1.0000008130183754348615718,
-1.0000007779252810548829833,
-1.0000007448231994552217961,
-1.0000007135729059221974924,
-1.0000006840466971318745829,
-1.0000006561272939441861046,
-1.0000006297068620102598893,
-1.0000006046861361493852810,
-1.0000005809736362878291024,
-1.0000005584849643282480472,
-1.0000005371421726703363639,
-1.0000005168731962740692049,
-1.0000004976113411596806008,
-1.0000004792948231127072342,
-1.0000004618663511196359830,
-1.0000004452727507103229274,
-1.0000004294646229590628680,
-1.0000004143960353902593347,
-1.0000004000242414690848708,
-1.0000003863094257375342469,
-1.0000003732144719883643755,
-1.0000003607047521588966840,
-1.0000003487479338864325415,
-1.0000003373138048878843165,
-1.0000003263741125275609889,
-1.0000003159024171116037866,
-1.0000003058739576006566652,
-1.0000002962655285725437904,
-1.0000002870553673827358140,
-1.0000002782230505849869057,
-1.0000002697493987622364559,
-1.0000002616163890110023357,
-1.0000002538070743900383910,
-1.0000002463055097187303796,
-1.0000002390966831671914941,
-1.0000002321664531315182148,
-1.0000002255014899446962720,
-1.0000002190892220043294189,
-1.0000002129177859488875363,
-1.0000002069759805409503856,
-1.0000002012532239508346352,
-1.0000001957395141607715799,
-1.0000001904253922358238246,
-1.0000001853019082297385073,
-1.0000001803605895158355218,
-1.0000001755934113488585510,
-1.0000001709927694851839236,
-1.0000001665514546978896088,
-1.0000001622626290424854581,
-1.0000001581198037385383656,
-1.0000001541168185448943406,
-1.0000001502478225146562651,
-1.0000001465072560282191724,
-1.0000001428898340098206168,
-1.0000001393905302394604986,
-1.0000001360045626810435851,
-1.0000001327273797537779215,
-1.0000001295546474781991350,
-1.0000001264822374353463674,
-1.0000001235062154811934418,
-1.0000001206228311637514566,
-1.0000001178285077943789713,
-1.0000001151198331268959291,
-1.0000001124935506041689973,
-1.0000001099465511333538870,
-1.0000001074758653540159800,
-1.0000001050786563654105762,
-1.0000001027522128845166638,
-1.0000001004939428029486687,
-1.0000000983013671212789805,
-1.0000000961721142318230568,
-1.0000000941039145299377849,
-1.0000000920945953319322191,
-1.0000000901420760810508342,
-1.0000000882443638214715567,
-1.0000000863995489252481624,
-1.0000000846058010559340087,
-1.0000000828613653539251108,
-1.0000000811645588297531945,
-1.0000000795137669539446035,
-1.0000000779074404293504316,
-1.0000000763440921371658435,
-1.0000000748222942452544601,
-1.0000000733406754682610490,
-1.0000000718979184725736242,
-1.0000000704927574156181957,
-1.0000000691239756129809549,
-1.0000000677904033262021616,
-1.0000000664909156620260150,
-1.0000000652244305800707413,
-1.0000000639899069999200210,
-1.0000000627863430035157863,
-1.0000000616127741272145396,
-1.0000000604682717384114402,
-1.0000000593519414925037722,
-1.0000000582629218640138419,
-1.0000000572003827506786799,
-1.0000000561635241443265987,
-1.0000000551515748656132565,
-1.0000000541637913591477838,
-1.0000000531994565462984639,
-1.0000000522578787309074810,
-1.0000000513383905583484138,
-1.0000000504403480213128433,
-1.0000000495631295122176546,
-1.0000000487061349200646268,
-1.0000000478687847678491846,
-1.0000000470505193895425292,
-1.0000000462507981442619037,
-1.0000000454690986663279506,
-1.0000000447049161481733945,
-1.0000000439577626562114715,
-1.0000000432271664758693969,
-1.0000000425126714863289720,
-1.0000000418138365622638258,
-1.0000000411302350023564495,
-1.0000000404614539811255780,
-1.0000000398070940262323231,
-1.0000000391667685166029872,
-1.0000000385401032031032834,
-1.0000000379267357488366139,
-1.0000000373263152893916691,
-1.0000000367385020107625221,
-1.0000000361629667461338400,
-1.0000000355993905882786080,
-1.0000000350474645183273051,
-1.0000000345068890502580120,
-1.0000000339773738891558860,
-1.0000000334586376040009459,
-1.0000000329504073138999626,
-1.0000000324524183859193141,
-1.0000000319644141456030054,
-1.0000000314861455999590131,
};

inline static double lega_seq_norm (size_t n) {
    if (n <= 200){
        return (double) legaseqnorm[n];
    }
    return ( sqrt( 4 * (double) (n * n) - 1) / (double) n);
}

inline static double legc_seq_norm (size_t n) {
    if (n <= 200){
        return (double) legcseqnorm[n];
    }
    return ( -(sqrt(2 * (double) n + 1) * ((double) n -1)) / (double)n / sqrt( 2 * (double) n - 3));}


// Orthonormality functions
double chebortho(size_t n) {
    if (n == 0){
        return M_PI;
    }
    else{
        return M_PI/2.0;
    }
}

/* static const double legorthoarr[201] = */
/*             {1.000000000000000e+00, 3.333333333333333e-01,2.000000000000000e-01, */
/*             1.428571428571428e-01,1.111111111111111e-01,9.090909090909091e-02,7.692307692307693e-02, */
/*             6.666666666666667e-02,5.882352941176471e-02,5.263157894736842e-02,4.761904761904762e-02, */
/*             4.347826086956522e-02,4.000000000000000e-02,3.703703703703703e-02,3.448275862068965e-02, */
/*             3.225806451612903e-02,3.030303030303030e-02,2.857142857142857e-02,2.702702702702703e-02, */
/*             2.564102564102564e-02,2.439024390243903e-02,2.325581395348837e-02,2.222222222222222e-02, */
/*             2.127659574468085e-02,2.040816326530612e-02,1.960784313725490e-02,1.886792452830189e-02, */
/*             1.818181818181818e-02,1.754385964912281e-02,1.694915254237288e-02,1.639344262295082e-02 */
/*             ,1.587301587301587e-02,1.538461538461539e-02,1.492537313432836e-02,1.449275362318841e-02 */
/*             ,1.408450704225352e-02,1.369863013698630e-02,1.333333333333333e-02,1.298701298701299e-02 */
/*             ,1.265822784810127e-02,1.234567901234568e-02,1.204819277108434e-02,1.176470588235294e-02 */
/*             ,1.149425287356322e-02,1.123595505617977e-02,1.098901098901099e-02,1.075268817204301e-02 */
/*             ,1.052631578947368e-02,1.030927835051546e-02,1.010101010101010e-02,9.900990099009901e-03 */
/*             ,9.708737864077669e-03,9.523809523809525e-03,9.345794392523364e-03,9.174311926605505e-03 */
/*             ,9.009009009009009e-03,8.849557522123894e-03,8.695652173913044e-03,8.547008547008548e-03 */
/*             ,8.403361344537815e-03,8.264462809917356e-03,8.130081300813009e-03,8.000000000000000e-03 */
/*             ,7.874015748031496e-03,7.751937984496124e-03,7.633587786259542e-03,7.518796992481203e-03 */
/*             ,7.407407407407408e-03,7.299270072992700e-03,7.194244604316547e-03,7.092198581560284e-03 */
/*             ,6.993006993006993e-03,6.896551724137931e-03,6.802721088435374e-03,6.711409395973154e-03 */
/*             ,6.622516556291391e-03,6.535947712418301e-03,6.451612903225806e-03,6.369426751592357e-03 */
/*             ,6.289308176100629e-03,6.211180124223602e-03,6.134969325153374e-03,6.060606060606061e-03 */
/*             ,5.988023952095809e-03,5.917159763313609e-03,5.847953216374269e-03,5.780346820809248e-03 */
/*             ,5.714285714285714e-03,5.649717514124294e-03,5.586592178770950e-03,5.524861878453038e-03 */
/*             ,5.464480874316940e-03,5.405405405405406e-03,5.347593582887700e-03,5.291005291005291e-03 */
/*             ,5.235602094240838e-03,5.181347150259068e-03,5.128205128205128e-03,5.076142131979695e-03 */
/*             ,5.025125628140704e-03,4.975124378109453e-03,4.926108374384237e-03,4.878048780487805e-03 */
/*             ,4.830917874396135e-03,4.784688995215311e-03,4.739336492890996e-03,4.694835680751174e-03 */
/*             ,4.651162790697674e-03,4.608294930875576e-03,4.566210045662100e-03,4.524886877828055e-03 */
/*             ,4.484304932735426e-03,4.444444444444444e-03,4.405286343612335e-03,4.366812227074236e-03 */
/*             ,4.329004329004329e-03,4.291845493562232e-03,4.255319148936170e-03,4.219409282700422e-03 */
/*             ,4.184100418410041e-03,4.149377593360996e-03,4.115226337448560e-03,4.081632653061225e-03 */
/*             ,4.048582995951417e-03,4.016064257028112e-03,3.984063745019920e-03,3.952569169960474e-03 */
/*             ,3.921568627450980e-03,3.891050583657588e-03,3.861003861003861e-03,3.831417624521073e-03 */
/*             ,3.802281368821293e-03,3.773584905660377e-03,3.745318352059925e-03,3.717472118959108e-03 */
/*             ,3.690036900369004e-03,3.663003663003663e-03,3.636363636363636e-03,3.610108303249098e-03 */
/*             ,3.584229390681004e-03,3.558718861209964e-03,3.533568904593640e-03,3.508771929824561e-03 */
/*             ,3.484320557491289e-03,3.460207612456748e-03,3.436426116838488e-03,3.412969283276451e-03 */
/*             ,3.389830508474576e-03,3.367003367003367e-03,3.344481605351171e-03,3.322259136212625e-03 */
/*             ,3.300330033003300e-03,3.278688524590164e-03,3.257328990228013e-03,3.236245954692557e-03 */
/*             ,3.215434083601286e-03,3.194888178913738e-03,3.174603174603175e-03,3.154574132492113e-03 */
/*             ,3.134796238244514e-03,3.115264797507788e-03,3.095975232198143e-03,3.076923076923077e-03 */
/*             ,3.058103975535168e-03,3.039513677811550e-03,3.021148036253776e-03,3.003003003003003e-03 */
/*             ,2.985074626865672e-03,2.967359050445104e-03,2.949852507374631e-03,2.932551319648094e-03 */
/*             ,2.915451895043732e-03,2.898550724637681e-03,2.881844380403458e-03,2.865329512893983e-03 */
/*             ,2.849002849002849e-03,2.832861189801700e-03,2.816901408450704e-03,2.801120448179272e-03 */
/*             ,2.785515320334262e-03,2.770083102493075e-03,2.754820936639119e-03,2.739726027397260e-03 */
/*             ,2.724795640326975e-03,2.710027100271003e-03,2.695417789757413e-03,2.680965147453083e-03 */
/*             ,2.666666666666667e-03,2.652519893899204e-03,2.638522427440633e-03,2.624671916010499e-03 */
/*             ,2.610966057441253e-03,2.597402597402597e-03,2.583979328165375e-03,2.570694087403599e-03 */
/*             ,2.557544757033248e-03,2.544529262086514e-03,2.531645569620253e-03,2.518891687657431e-03 */
/*             ,2.506265664160401e-03,2.493765586034913e-03}; */

// Helper functions
double orth_poly_expansion_eval2(double x, void * p){
    
    struct OrthPolyExpansion * temp = p;
    return orth_poly_expansion_eval(temp,x);
}

double orth_poly_expansion_eval3(double x, void * p)
{
    struct OrthPolyExpansion ** temp = p;
    double out = orth_poly_expansion_eval(temp[0],x);
    out *= orth_poly_expansion_eval(temp[1],x);

    return out;
}

struct lin_func
{
    double slope;
    double offset;
};

double eval_lin_func(double x, void * args)
{
    struct lin_func * lf = args;
    double m = lf->slope;
    double b = lf->offset;
    return m*x + b;
}

struct quad_func
{
    double scale;
    double offset;
};

double eval_quad_func(double x, void * args)
{
    struct quad_func * qf = args;
    double m = qf->scale;
    double b = qf->offset;
    return m*(x-b)*(x-b);
}

struct OpeOpts{
    enum poly_type ptype;
    
    size_t start_num;
    size_t max_num;
    size_t coeffs_check;
    double tol;
    
    double lb;
    double ub;

    // for hermite
    double mean;
    double std;


    // kristoffel weighting for least squares
    int kristoffel_eval;
    enum quad_rule qrule;

};

struct OpeOpts * ope_opts_alloc(enum poly_type ptype)
{
    struct OpeOpts * ao;
    if ( NULL == (ao = malloc(sizeof(struct OpeOpts)))){
        fprintf(stderr, "failed to allocate memory for struct OpeOpts in ope_opts_alloc.\n");
        exit(1);
    }

    ao->start_num = 5;
    ao->coeffs_check = 2;
    ao->tol = 1e-10;
    ao->max_num = 100;

    ao->ptype = ptype;
    if (ptype == HERMITE){
        ao->lb = -DBL_MAX;
        ao->ub = DBL_MAX;
    }
    else if (ptype == FOURIER){
        ao->lb = 0.0;
        ao->ub = 2.0*M_PI;
        ao->start_num = 8;
    }
    else{
        ao->lb = -1.0;
        ao->ub = 1.0;
    }
    

    // for hermite
    ao->mean = 0.0;
    ao->std  = 1.0;


    // for least squares applications
    ao->kristoffel_eval = 0;
    ao->qrule = C3_GAUSS_QUAD;

    return ao;
}

void ope_opts_free(struct OpeOpts * ope)
{
    if (ope != NULL){
        free(ope); ope = NULL;
    }
}

void ope_opts_free_deep(struct OpeOpts ** ope)
{
    if (*ope != NULL){
        free(*ope); *ope = NULL;
    }
}

void ope_opts_set_start(struct OpeOpts * ope, size_t start)
{
    assert (ope != NULL);
    ope->start_num = start;
}

size_t ope_opts_get_start(const struct OpeOpts * ope)
{
    assert (ope != NULL);
    return ope->start_num;
}

void ope_opts_set_maxnum(struct OpeOpts * ope, size_t maxnum)
{
    assert (ope != NULL);
    ope->max_num = maxnum;
}

size_t ope_opts_get_maxnum(const struct OpeOpts * ope)
{
    assert (ope != NULL);
    assert (ope->max_num < 1000); // really a check if it exists
    
    return ope->max_num;
}

void ope_opts_set_coeffs_check(struct OpeOpts * ope, size_t num)
{
    assert (ope != NULL);
    ope->coeffs_check = num;
}

void ope_opts_set_tol(struct OpeOpts * ope, double tol)
{
    assert (ope != NULL);
    ope->tol = tol;
}

void ope_opts_set_lb(struct OpeOpts * ope, double lb)
{
    assert (ope != NULL);
    ope->lb = lb;
}

double ope_opts_get_lb(const struct OpeOpts * ope)
{
    assert (ope != NULL);
    return ope->lb;
}

void ope_opts_set_ub(struct OpeOpts * ope, double ub)
{
    assert (ope != NULL);
    ope->ub = ub;
}

double ope_opts_get_ub(const struct OpeOpts * ope)
{
    assert (ope != NULL);
    return ope->ub;
}

void ope_opts_set_mean_and_std(struct OpeOpts * ope, double mean, double std)
{
    if (ope->ptype == HERMITE){
        ope->mean = mean;
        ope->std = std;
    }
    else{
        fprintf(stderr,"Warning: setting mean and variance for non hermite polynomials\n");
        fprintf(stderr,"         does not do anything\n");
    }

}

void ope_opts_set_ptype(struct OpeOpts * ope, enum poly_type ptype)
{
    assert (ope != NULL);
    ope->ptype = ptype;
}

enum poly_type ope_opts_get_ptype(const struct OpeOpts * ope)
{
    assert (ope != NULL);
    return ope->ptype;
}

void ope_opts_set_qrule(struct OpeOpts * ope, enum quad_rule qrule)
{
    assert (ope != NULL);
    if ((qrule != C3_GAUSS_QUAD) && (qrule != C3_CC_QUAD)){
        fprintf(stderr,"Specified qrule %d is not known\n",qrule);
        exit(1);
    }
    ope->qrule = qrule;
}

/********************************************************//**
*   Get number of free parameters
*************************************************************/
size_t ope_opts_get_nparams(const struct OpeOpts * opts)
{
    assert (opts != NULL);
    return opts->start_num;
}

/********************************************************//**
*   Set number of free parameters
*************************************************************/
void ope_opts_set_nparams(struct OpeOpts * opts, size_t num)
{
    assert (opts != NULL);
    opts->start_num = num;
}


/********************************************************//**
*   Set kristoffel weighting for evaluation of polynomials
*   This is typically only used in the context of least squares
*  
*   \param[in] opts              - options to modify
*   \param[in] kristoffel_weight - 1 to use, 0 to not use
*************************************************************/
void ope_opts_set_kristoffel_weight(struct OpeOpts * opts, int kristoffel_weight)
{
    assert (opts != NULL);
    opts->kristoffel_eval = kristoffel_weight;
}

/********************************************************//**
*   Initialize a standard basis polynomial
*
*   \param[in] num_poly - number of basis
*   \param[in] lb       - lower bound
*   \param[in] ub       - upper bound
*
*   \return  standard polynomial
*************************************************************/
struct StandardPoly * 
standard_poly_init(size_t num_poly, double lb, double ub){
    
    struct StandardPoly * p;
    if ( NULL == (p = malloc(sizeof(struct StandardPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = STANDARD;
    p->num_poly = num_poly;
    p->lower_bound = lb;
    p->upper_bound = ub;
    p->coeff = calloc_double(num_poly);
    return p;
}

/********************************************************//**
*   Create the polynomial representing the derivative
*   of the standard polynomial
*
*   \param[in] p - polynomial
*
*   \return  derivative polynomial
*************************************************************/
struct StandardPoly * 
standard_poly_deriv(struct StandardPoly * p){
    
    struct StandardPoly * dp;
    if ( NULL == (dp = malloc(sizeof(struct StandardPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    dp->ptype = STANDARD;
    dp->lower_bound = p->lower_bound;
    dp->upper_bound = p->upper_bound;
    if (p->num_poly > 1){
        dp->num_poly = p->num_poly-1;
        dp->coeff = calloc_double(dp->num_poly);
        size_t ii;
        for (ii = 1; ii < p->num_poly; ii++){
            dp->coeff[ii-1] = p->coeff[ii] * (double) ii;
        }
    }
    else{
        dp->num_poly = 1;
        dp->coeff = calloc_double(dp->num_poly);
    }

    return dp;
}

/********************************************************//**
*   free memory allocated to a standard polynomial
*
*   \param[in,out] p - polynomial structure 
*
*************************************************************/
void 
standard_poly_free(struct StandardPoly * p)
{
    free(p->coeff);
    free(p);
}

/********************************************************//**
*   Initialize a Chebyshev polynomial 
*
*   \return p - polynomial
*************************************************************/
struct OrthPoly * init_cheb_poly(){
    
    struct OrthPoly * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = CHEBYSHEV;
    p->an = &two_seq; 
    p->bn = &zero_seq;
    p->cn = &none_seq;
    
    p->lower = -1.0;
    p->upper = 1.0;

    p->const_term = 1.0;
    p->lin_coeff = 1.0;
    p->lin_const = 0.0;

    p->norm = &chebortho;

    return p;
}

inline static double legortho(size_t n){
    (void) n;
    return 1;
    /* if (n < 201){ */
    /*     return legorthoarr[n]; */
    /* } */
    /* else{ */
    /*    // printf("here?! n=%zu\n",n); */
    /*     return (1.0 / (2.0 * (double) n + 1.0)); */
    /* } */
}

/********************************************************//**
*   Initialize a Legendre polynomial 
*
*   \return p - polynomial
*************************************************************/
struct OrthPoly * init_leg_poly(){
    
    struct OrthPoly * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPoly)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    p->ptype = LEGENDRE;
    p->bn = &zero_seq;


    /* orthogonal */
    /* p->an = &lega_seq; */
    /* p->cn = &legc_seq; */
    /* p->lin_coeff = 1.0;  */

    /* orthonormal */
    p->an = &lega_seq_norm;
    p->cn = &legc_seq_norm;
    p->lin_coeff = 1.0 * sqrt(3.0);
    
    p->lower = -1.0;
    p->upper = 1.0;

    p->const_term = 1.0;


    p->lin_const = 0.0;

    p->norm = &legortho;

    return p;
}

/********************************************************//**
*   free memory allocated to a polynomial
*
*   \param[in,out] p - polynomial structure to free
*************************************************************/
void free_orth_poly(struct OrthPoly * p)
{
    if (p != NULL){
        free(p); p = NULL;
    }
}

/********************************************************//**
*   serialize an orthonormal polynomial
*
*   \param p - polynomial structure to serialize
*
*   \return ser - serialize polynomial
*
*   \note 
*    This is actually pretty stupid because only need to
*    serialize the type. But hey. Its good practice.
*************************************************************/
unsigned char *
serialize_orth_poly(struct OrthPoly * p)
{
    
    /*
    char start[]= "type=";
    char * ser = NULL;
    concat_string_ow(&ser,start);
    
    char temp[3];
    snprintf(temp,2,"%d",p->ptype);
    concat_string_ow(&ser,temp);
    */
    
    unsigned char * ser = malloc(sizeof(int) * sizeof(unsigned char)) ;
    serialize_int(ser, p->ptype);
    return ser;
}

/********************************************************//**
*   deserialize an orthonormal polynomial
*
*   \param[in] ser - string to deserialize
*
*   \return poly - orthonormal polynomial
*************************************************************/
struct OrthPoly *
deserialize_orth_poly(unsigned char * ser)
{

    struct OrthPoly * poly = NULL;
    
    int type;
    deserialize_int(ser, &type);
    if (type == LEGENDRE) {
        poly = init_leg_poly();
    }
    else if (type == HERMITE){
        poly = init_hermite_poly();
    }
    else if (type == CHEBYSHEV){
        poly = init_cheb_poly();
    }
    else if (type == FOURIER){
        poly = init_fourier_poly();
    }
    else{
        fprintf(stderr,"Cannot desrialize polynomial of type %d\n",type);
    }
    return poly;
}

/********************************************************//**
*   Convert an orthogonal family polynomial of order *n*
*   to a standard_polynomial
*
*   \param[in] p - polynomial
*   \param[in] n - polynomial order
*
*   \return sp - standard polynomial
*************************************************************/
struct StandardPoly * orth_to_standard_poly(struct OrthPoly * p, size_t n)
{
    struct StandardPoly * sp = standard_poly_init(n+1,p->lower,p->upper);
    size_t ii, jj;
    if (n == 0){
        sp->coeff[n] = p->const_term;
    }
    else if (n == 1){
        sp->coeff[0] = p->lin_const;
        sp->coeff[1] = p->lin_coeff;
    }
    else{
        
        double * a = calloc_double(n+1); //n-2 poly
        a[0] = p->const_term;
        double * b = calloc_double(n+1); // n- 1poly
        b[0] = p->lin_const;
        b[1] = p->lin_coeff;
        for (ii = 2; ii < n+1; ii++){
            sp->coeff[0] = p->bn(ii) * b[0] + p->cn(ii) * a[0];
            for (jj = 1; jj < ii-1; jj++){
                sp->coeff[jj] = p->an(ii)*b[jj-1] + p->bn(ii) * b[jj] + 
                                    p->cn(ii) * a[jj];
            }
            sp->coeff[ii-1] = p->an(ii)*b[ii-2] + p->bn(ii) * b[ii-1];
            sp->coeff[ii] = p->an(ii) * b[ii-1];

            memcpy(a, b, ii * sizeof(double));
            memcpy(b, sp->coeff, (ii+1) * sizeof(double));
        }
        
        free(a);
        free(b);
    }

    return sp;
}

/********************************************************//**
*   Evaluate an orthogonal polynomial with previous two
*   polynomial orders specified
*
*   \param[in] rec - orthogonal polynomial 
*   \param[in] p2  - if evaluating polynomial P_n(x), then p2 is P_{n-2}(x)
*   \param[in] p1  - if evaluating polynomial P_n(x), then p1 is P_{n-1}(x)
*   \param[in] n   - order
*   \param[in] x   - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
inline static double 
eval_orth_poly_wp(const struct OrthPoly * rec, double p2, double p1, 
                  size_t n, double x)
{   
    return (rec->an(n) * x + rec->bn(n)) * p1 + rec->cn(n) * p2; 
}

/********************************************************//**
*   Evaluate the derivative of a legendre polynomial up to a certain
*   order
*
*   \param[in] x     - location at which to evaluate
*   \param[in] order - maximum order;
*
*   \return out - derivatives
*************************************************************/
double * deriv_legen_upto(double x, size_t order){
    
    double * out = calloc_double(order+1);
    if (order == 0){
        return out;   
    }
    else if( fabs(x-1.0) <= DBL_EPSILON) {
        size_t ii;
        for (ii = 1; ii < order+1; ii++){
            out[ii] = (double) (order * (order+1)/2.0);
            
            out[ii] *= sqrt(2*ii+1); // for orthonormal
        }
    }
    else if (fabs(x-(-1.0)) <= DBL_EPSILON){
        size_t ii;
        for (ii = 1; ii < order+1; ii+=2){
            out[ii] = (double) (order * (order+1)/2.0);
            out[ii] *= sqrt(2*ii+1); // for orthonormal
        }
        for (ii = 2; ii < order+1; ii+=2){
            out[ii] = -(double) (order * (order+1)/2.0);
            out[ii] *= sqrt(2*ii+1); // for orthonormal
        }
    }
    else if (order == 1){
        struct OrthPoly * p = init_leg_poly();
        out[1] = x * orth_poly_eval(p,order,x) - orth_poly_eval(p,order-1,x);
        out[1] = (double) order * out[1] / ( x * x - 1.0);
        free_orth_poly(p);
    }
    else{
        struct OrthPoly * p = init_leg_poly();
        double eval0 = orth_poly_eval(p,0,x);
        double eval1 = orth_poly_eval(p,1,x);
        double evaltemp;

        out[1] = x * eval1 - eval0;
        //printf("out[1]=%G\n",out[1]);
        out[1] = 1.0  * out[1] / ( x * x - 1.0);
        
        size_t ii;
        for (ii = 2; ii < order+1; ii++){
            evaltemp = eval_orth_poly_wp(p, eval0, eval1, ii, x);
            eval0 = eval1;
            eval1 = evaltemp;
            out[ii] = x * eval1 - eval0;
            out[ii] = (double) ii * out[ii] / ( x * x - 1.0);
        }
        free_orth_poly(p);
    }

    return out;
}

/********************************************************//**
*   Evaluate the derivative of a legendre polynomial of a certain order
*
*   \param[in] x     - location at which to evaluate
*   \param[in] order - order of the polynomial
*
*   \return out - derivative
*************************************************************/
double deriv_legen(double x, size_t order){
    
    if (order == 0){
        return 0.0;
    }
    double out;

    if ( fabs(x-1.0) <= DBL_EPSILON) {
        out = (double) (order * (order+1)/2.0);
    }
    else if (fabs(x-(-1.0)) <= DBL_EPSILON){
        if (order % 2){ // odd
            out = (double) (order * (order+1)/2.0);
        }
        else{
            out = -(double) (order * (order+1)/2.0);
        }
    }
    else{
        struct OrthPoly * p = init_leg_poly();
        out = x * orth_poly_eval(p,order,x) - orth_poly_eval(p,order-1,x);
        //printf("out in plain = %G\n",out);
        out = (double) order * out / ( x * x - 1.0);
        free_orth_poly(p);
    }
    return out;
}

/********************************************************//**
*   Evaluate the derivative of an orthogonal polynomial
*
*   \param[in] ptype - polynomial type
*   \param[in] order - order of the polynomial
*   \param[in] x     - location at which to evaluate
*
*   \return out - orthonormal polynomial expansion 
*************************************************************/
double orth_poly_deriv(enum poly_type ptype, size_t order, double x)
{
    assert (1 == 0);
    double out = 0.0;
    if (ptype == LEGENDRE){
        out = deriv_legen(x,order);
    }
    else {
        fprintf(stderr,"Have not implemented orth_poly_deriv for %d\n",ptype);
        exit(1);
    }
    return out;
}

/********************************************************//**
*   Evaluate an orthogonal polynomial of a given order
*
*   \param[in] rec - orthogonal polynomial 
*   \param[in] n   - order
*   \param[in] x   - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double 
orth_poly_eval(const struct OrthPoly * rec, size_t n, double x)
{   
    if (n == 0){
        return rec->const_term;
    }
    else if (n == 1){
        return rec->lin_coeff * x + rec->lin_const;
    }
    else {
        double out = (rec->an(n)*x + rec->bn(n)) * orth_poly_eval(rec,n-1,x) +
                        rec->cn(n) * orth_poly_eval(rec,n-2,x);
        return out;
    }
}

/********************************************************//**
*   Get number of polynomials
*************************************************************/
size_t orth_poly_expansion_get_num_poly(const struct OrthPolyExpansion * ope)
{
    assert (ope != NULL);
    return ope->num_poly;
}

/********************************************************//**
*   Get number of parameters
*************************************************************/
size_t orth_poly_expansion_get_num_params(const struct OrthPolyExpansion * ope)
{
    assert (ope != NULL);
    return ope->num_poly;
}

/********************************************************//**
*   Get lower bounds
*************************************************************/
double orth_poly_expansion_get_lb(const struct OrthPolyExpansion * ope)
{
    assert (ope != NULL);
    return ope->lower_bound;
}

/********************************************************//**
*   Get upper bounds
*************************************************************/
double orth_poly_expansion_get_ub(const struct OrthPolyExpansion * ope)
{
    assert (ope != NULL);
    return ope->upper_bound;
}


/********************************************************//**
*   Initialize an expansion of a certain orthogonal polynomial family
*            
*   \param[in] ptype    - type of polynomial
*   \param[in] num_poly - number of polynomials
*   \param[in] lb       - lower bound
*   \param[in] ub       - upper bound
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_init(enum poly_type ptype, size_t num_poly, 
                         double lb, double ub)
{

    struct OrthPolyExpansion * p;
    if ( NULL == (p = malloc(sizeof(struct OrthPolyExpansion)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }

    p->num_poly = num_poly;

    //p->coeff = calloc_double(num_poly);
    p->lower_bound = lb;
    p->upper_bound = ub;
    p->kristoffel_eval = 0;
    
    double m, off;
    switch (ptype) {
        case LEGENDRE:
            p->p = init_leg_poly();
            p->space_transform = space_mapping_create(SM_LINEAR);
            m = (p->p->upper - p->p->lower) /  (ub - lb);
            off = p->p->upper - m * ub;
            p->space_transform->set = 1;
            p->space_transform->lin_slope = m;
            p->space_transform->lin_offset = off;
            p->space_transform->inv_lin_slope = 1.0/m;
            p->space_transform->inv_lin_offset = -off/m;
            break;
        case CHEBYSHEV:
            p->p = init_cheb_poly();
            p->space_transform = space_mapping_create(SM_LINEAR);
            m = (p->p->upper - p->p->lower) /  (ub - lb);
            off = p->p->upper - m * ub;
            p->space_transform->set = 1;
            p->space_transform->lin_slope = m;
            p->space_transform->lin_offset = off;
            p->space_transform->inv_lin_slope = 1.0/m;
            p->space_transform->inv_lin_offset = -off/m;
            break;
        case FOURIER:
            p->p = init_fourier_poly();
            p->space_transform = space_mapping_create(SM_LINEAR);
            m = (p->p->upper - p->p->lower) /  (ub - lb);
            off = p->p->upper - m * ub;
            p->space_transform->set = 1;
            p->space_transform->lin_slope = m;
            p->space_transform->lin_offset = off;
            p->space_transform->inv_lin_slope = 1.0/m;
            p->space_transform->inv_lin_offset = -off/m;
            break;            
        case HERMITE:
            p->p = init_hermite_poly();
            p->space_transform = space_mapping_create(SM_LINEAR);;
            break; 
        case STANDARD:
            p->space_transform = space_mapping_create(SM_LINEAR);
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", ptype);
    }

    p->nalloc = num_poly+OPECALLOC;
    p->coeff = calloc_double(p->nalloc);
    p->ccoeff = NULL;
    if (ptype == FOURIER){
        p->ccoeff = malloc(p->nalloc * sizeof(double complex));
        for (size_t ii = 0; ii < p->nalloc; ii++){
            p->ccoeff[ii] = 0.0;
        }
    }

    return p;
}

/********************************************************//**
*   Initialize an expansion of a certain orthogonal polynomial family
*            
*   \param[in] opts     - approximation options
*   \param[in] num_poly - number of polynomials
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_init_from_opts(const struct OpeOpts * opts, size_t num_poly)
{

    struct OrthPolyExpansion * p =
        orth_poly_expansion_init(opts->ptype,num_poly,opts->lb,opts->ub);
    if (opts->ptype == HERMITE)
    {
        p->space_transform->set = 1;
        p->space_transform->lin_slope = 1.0/opts->std;
        p->space_transform->lin_offset = -opts->mean/opts->std;
        p->space_transform->inv_lin_slope = opts->std;
        p->space_transform->inv_lin_offset = opts->mean;
    }

    p->kristoffel_eval = opts->kristoffel_eval;

    return p;
}

/********************************************************//**
*   Initialize an expanion of a certain orthogonal polynomial family
*            
*   \param[in] opts    - approximation options
*   \param[in] nparams - number of polynomials
*   \param[in] param   - parameters
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_create_with_params(struct OpeOpts * opts,
                                       size_t nparams, const double * param)
{

    struct OrthPolyExpansion * poly = NULL;
    poly = orth_poly_expansion_init_from_opts(opts,nparams);
    for (size_t ii = 0; ii < nparams; ii++){
        poly->coeff[ii] = param[ii];
    }
    return poly;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
size_t orth_poly_expansion_get_params(const struct OrthPolyExpansion * ope, double * param)
{
    assert (ope != NULL);
    memmove(param,ope->coeff,ope->num_poly * sizeof(double));
    return ope->num_poly;
}

/********************************************************//**
*   Get parameters defining polynomial (for now just coefficients)
*************************************************************/
double * orth_poly_expansion_get_params_ref(
    const struct OrthPolyExpansion * ope, size_t *nparam)
{
    assert (ope != NULL);
    *nparam = ope->num_poly;
    return ope->coeff;
}

/********************************************************//**
*   Update an expansion's parameters
*            
*   \param[in] ope     - expansion to update
*   \param[in] nparams - number of polynomials
*   \param[in] param   - parameters

*   \returns 0 if successful
*************************************************************/
int
orth_poly_expansion_update_params(struct OrthPolyExpansion * ope,
                                  size_t nparams, const double * param)
{

    size_t nold = ope->num_poly;
    ope->num_poly = nparams;
    if (nold >= nparams){
        for (size_t ii = 0; ii < nparams; ii++){
            ope->coeff[ii] = param[ii];
        }
        for (size_t ii = nparams; ii < nold; ii++){
            ope->coeff[ii] = 0.0;
        }
    }
    else{
        if (nparams <= ope->nalloc){

            for (size_t ii = 0; ii < nparams; ii++){
                ope->coeff[ii] = param[ii];
            }
        }
        else{
            free(ope->coeff); ope->coeff = NULL;
            ope->nalloc = nparams;
            ope->coeff = calloc_double(ope->nalloc);
            for (size_t ii = 0; ii < nparams; ii++){
                ope->coeff[ii] = param[ii];
            }
        }
    }
    return 0;
}

/********************************************************//**
*   Copy an orthogonal polynomial expansion
*            
*   \param[in] pin - polynomial to copy
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * orth_poly_expansion_copy(const struct OrthPolyExpansion * pin)
{
    struct OrthPolyExpansion * p = NULL;
    if (pin != NULL){
        assert (pin->nalloc >= pin->num_poly);

        if ( NULL == (p = malloc(sizeof(struct OrthPolyExpansion)))){
            fprintf(stderr, "failed to allocate memory for poly exp.\n");
            exit(1);
        }
        //   printf("copying polynomial\n");
        //printf("pin->num_poly = %zu, pin->nalloc = %zu\n",pin->num_poly,pin->nalloc);
        p->num_poly = pin->num_poly;
        p->nalloc = pin->nalloc;
        p->coeff = NULL;
        p->ccoeff = NULL;
        p->lower_bound = pin->lower_bound;
        p->upper_bound = pin->upper_bound;
        p->space_transform = space_mapping_copy(pin->space_transform);
        p->kristoffel_eval = pin->kristoffel_eval;
        
        switch (pin->p->ptype) {
        case LEGENDRE:
            p->coeff = calloc_double(pin->nalloc);
            memmove(p->coeff,pin->coeff, p->num_poly * sizeof(double));
            p->p = init_leg_poly();
            break;
        case CHEBYSHEV:
            p->coeff = calloc_double(pin->nalloc);
            memmove(p->coeff,pin->coeff, p->num_poly * sizeof(double));
            p->p = init_cheb_poly();
            break;
        case HERMITE:
            p->coeff = calloc_double(pin->nalloc);
            memmove(p->coeff,pin->coeff, p->num_poly * sizeof(double));
            p->p = init_hermite_poly();
            break;
        case FOURIER:
            p->p = init_fourier_poly();
            p->ccoeff = malloc(p->nalloc * sizeof(double complex));
            for (size_t ii = 0; ii < p->nalloc; ii++){
                p->ccoeff[ii] = 0.0;
            }
            for (size_t ii = 0; ii < p->num_poly; ii++){
                p->ccoeff[ii] = pin->ccoeff[ii];
            }
            break;            
        case STANDARD:
            break;
            //default:
            //    fprintf(stderr, "Polynomial type does not exist: %d\n ", ptype);
        }
    

    }
    return p;
}

enum poly_type 
orth_poly_expansion_get_ptype(const struct OrthPolyExpansion * ope)
{
    assert (ope != NULL);
    return ope->p->ptype;
}

/********************************************************//**
    Return a zero function

    \param[in] opts         - extra arguments depending on function_class, sub_type, etc.
    \param[in] force_nparam - if == 1 then approximation will have the number of parameters
                                      defined by *get_nparams, for each approximation type
                              if == 0 then it may be more compressed

    \return p - zero function
************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_zero(struct OpeOpts * opts, int force_nparam)
{
    assert (opts != NULL);
    struct OrthPolyExpansion * p = NULL;
    
    if (opts->ptype == FOURIER){
        p = orth_poly_expansion_init_from_opts(opts, ope_opts_get_start(opts));
        return p;
    }
    

    if (force_nparam == 0){
        p = orth_poly_expansion_init_from_opts(opts,1);
        p->coeff[0] = 0.0;
    }
    else{
        size_t nparams = ope_opts_get_nparams(opts);
        p = orth_poly_expansion_init_from_opts(opts,nparams);
        for (size_t ii = 0; ii < nparams; ii++){
            p->coeff[ii] = 0.0;
        }
    }

    return p;
}


/********************************************************//**
*   Generate a constant orthonormal polynomial expansion
*
*   \param[in] a    - value of the function
*   \param[in] opts - opts
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_constant(double a, struct OpeOpts * opts)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (fabs(a) < 1e100);
    if (fabs(a) < ZEROTHRESH){
        return orth_poly_expansion_zero(opts, 0);
    }

    assert (opts != NULL);
  
  
    struct OrthPolyExpansion * p = NULL;
    if (opts->ptype != FOURIER){
        p = orth_poly_expansion_init_from_opts(opts,1);
        p->coeff[0] = a / p->p->const_term;
    }
    else{
        p = orth_poly_expansion_init_from_opts(opts,ope_opts_get_start(opts));
        p->ccoeff[0] = a;
    }



    return p;
}

/********************************************************//**
*   Generate a linear orthonormal polynomial expansion
*
*   \param[in] a      - value of the slope function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return p - orthogonal polynomial expansion
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_linear(double a, double offset, struct OpeOpts * opts)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);

    if (opts->ptype == FOURIER){
        fprintf(stderr,
                "Cannot initialize linear function for fourier basis\n");
        return NULL;
    }
    struct OrthPolyExpansion * p =
        orth_poly_expansion_init_from_opts(opts,2);
    p->coeff[1] = a / (p->p->lin_coeff * p->space_transform->lin_slope);
    p->coeff[0] = (offset - p->p->lin_const -
                   p->coeff[1] * p->p->lin_coeff *
                   p->space_transform->lin_offset) /
                   p->p->const_term;

    return p;
}

/********************************************************//**
*   Update a linear orthonormal polynomial expansion
*
*   \param[in] p      - existing linear polynomial
*   \param[in] a      - value of the slope function
*   \param[in] offset - offset
*
*   \return 0 if succesfull, 1 otherwise
*
*************************************************************/
int
orth_poly_expansion_linear_update(struct OrthPolyExpansion * p, double a, double offset)
{
    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);

    if (p->p->ptype == FOURIER){
        fprintf(stderr,
                "Cannot update linear function for fourier basis\n");
        return 1;
    }
    
    p->coeff[1] = a / (p->p->lin_coeff * p->space_transform->lin_slope);
    p->coeff[0] = (offset - p->p->lin_const -
                   p->coeff[1] * p->p->lin_coeff * p->space_transform->lin_offset) /
                   p->p->const_term;

    return 0;
}

/********************************************************//**
*   Generate a quadratic orthonormal polynomial expansion
    a * (x-offset)^2
*
*   \param[in] a      - value of the slope function
*   \param[in] offset - offset
*   \param[in] opts   - options
*
*   \return quadratic polynomial
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_quadratic(double a, double offset, struct OpeOpts * opts)
{

    assert (isnan(a) == 0);
    assert (isinf(a) == 0);
    assert (isnan(offset) == 0);
    assert (isinf(offset) == 0);
    
    if (opts->ptype == FOURIER){
        fprintf(stderr, "Cannot initialize quadratic function for fourier basis\n");
        return NULL;
    }
    
    struct OrthPolyExpansion * p = orth_poly_expansion_init_from_opts(opts, 3);

    struct quad_func qf;
    qf.scale = a;
    qf.offset = offset;
    orth_poly_expansion_approx(eval_quad_func, &qf, p);

    return p;
}

/********************************************************//**
*   Generate a polynomial expansion with only the
*   *order* coefficient being nonzero
*
*   \param[in] order - order of the polynomial
*   \param[in] opts  - options for building polynomial
*
*   \return p - orthogonal polynomial expansion
*
*   \note Not sure about hermite! 
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_genorder(size_t order, struct OpeOpts * opts)
{

    struct OrthPolyExpansion * p = 
        orth_poly_expansion_init_from_opts(opts, order+1);

    double m = space_mapping_map_inverse_deriv(p->space_transform,0);
    switch (opts->ptype){
    case LEGENDRE:
        p->coeff[order] = 1.0 / sqrt(p->p->norm(order)) / sqrt(2.0) / sqrt(m); 
        break;
    case HERMITE:
        p->coeff[order] = 1.0 / sqrt(p->p->norm(order)) / sqrt(m);
        break;
    case CHEBYSHEV:
        p->coeff[order] = 1.0 / sqrt(p->p->norm(order)) / sqrt(m);
        /* double norm = orth_poly_expansion_inner(p, p); */
        /* p->coeff[order] /= (norm * sqrt(m)); */
        /* break; */
        /* fprintf(stderr,"cannot generate orthonormal polynomial of\n"); */
        /* fprintf(stderr,"order for CHEBYSHEV type\n;"); */
        /* exit(1); */
        break;
    case FOURIER:
        /* p->coeff[order] = 1.0 / sqrt(p->p->norm(order)) / sqrt(m); */
        /* break; */
        fprintf(stderr,"cannot generate orthonormal polynomial of certin\n");
        fprintf(stderr,"order for FOURIER type\n;");
        exit(1);
    case STANDARD:
        fprintf(stderr,"cannot generate orthonormal polynomial of certin\n");
        fprintf(stderr,"order for STANDARD type\n;");
        exit(1);
    }
    //printf("wherhhh\n");
    

    //printf("there \n");
    return p;
}

/********************************************************//**
    Generate an orthonormal basis
    
    \param[in]     n    - number of basis function
    \param[in,out] f    - space to write polynomials
    \param[in]     opts - approximation options

    \note
    Uses modified gram schmidt to determine function coefficients
    Each function f[ii] must have the same nodes
*************************************************************/
void
orth_poly_expansion_orth_basis(size_t n, struct OrthPolyExpansion ** f,
                               struct OpeOpts * opts)
{

    if (opts->ptype == CHEBYSHEV){
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = orth_poly_expansion_init_from_opts(opts, ii+1);
            f[ii]->coeff[ii] = 1.0;
        }
        // now gram schmidt
        double norm, proj;
        for (size_t ii = 0; ii < n; ii++){
            norm = sqrt(orth_poly_expansion_inner(f[ii], f[ii])); 
            if (norm > 1e-200){
                orth_poly_expansion_scale(1.0/norm, f[ii]);
                for (size_t jj = ii+1; jj < n; jj++){
                    proj = orth_poly_expansion_inner(f[ii],f[jj]);
                    orth_poly_expansion_axpy(-proj,f[ii],f[jj]);
                }
            }
        }
    }
    else if ((opts->ptype == HERMITE)  || (opts->ptype == LEGENDRE)){
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = orth_poly_expansion_init_from_opts(opts, ii+1);
            double m = space_mapping_map_inverse_deriv(f[ii]->space_transform,0);
            f[ii]->coeff[ii] = 1.0 / sqrt(f[ii]->p->norm(ii)) / sqrt(m);
            if (opts->ptype == LEGENDRE){
                f[ii]->coeff[ii] /= sqrt(2.0);
            }
        }
    }
    else if (opts->ptype == FOURIER) {
        size_t N = ope_opts_get_start(opts);

        if (n > N){
            fprintf(stderr, "Cannot look for a rank so large for fourier\n");

            /* printf("n = %zu\n", n); */
            /* N = N-1; */
            /* while (n > N){ */
            /*     N = 2 * N; */
            /* } */
            /* N = N+1; */
            /* printf("N = %zu\n", N); */
        }
        for (size_t ii = 0; ii < n; ii++){
            f[ii] = orth_poly_expansion_init_from_opts(opts, N);
            if (n < N){
                f[ii]->ccoeff[ii] = 1.0;
            }
        }
        // now gram schmidt
        double norm, proj;
        for (size_t ii = 0; ii < n; ii++){
            /* printf("ii = %zu %zu\n", ii, f[ii]->num_poly); */
            norm = sqrt(orth_poly_expansion_inner(f[ii], f[ii])); 
            if (norm > 1e-200){
                orth_poly_expansion_scale(1.0/norm, f[ii]);
                for (size_t jj = ii+1; jj < n; jj++){
                    /* printf("jj = %zu before %zu\n", jj, f[jj]->num_poly); */
                    proj = orth_poly_expansion_inner(f[ii],f[jj]);
                    /* printf("jj = %zu after 1 %zu\n", jj, f[jj]->num_poly); */
                    orth_poly_expansion_axpy(-proj,f[ii],f[jj]);
                    /* printf("jj = %zu after  2%zu\n", jj, f[jj]->num_poly); */
                    /* if (f[jj]->num_poly != 33){ */
                    /*     printf("WARNING\n"); */
                    /*     exit(1); */
                    /* } */
                }
            }
        }
    }
    else{
        fprintf(stderr, "Cannot generate an orthonormal basis for polytype %d\n", opts->ptype);
        exit(1);
    }
}


/********************************************************//**
*   Evaluate the derivative of an orthogonal polynomial expansion
*
*   \param[in] poly - pointer to orth poly expansion
*   \param[in] x    - location at which to evaluate
*
*
*   \return out - value of derivative
*************************************************************/
double orth_poly_expansion_deriv_eval(const struct OrthPolyExpansion * poly, double x)
{
    assert (poly != NULL);
    assert (poly->kristoffel_eval == 0);

    if (poly->p->ptype == FOURIER){
        return fourier_expansion_deriv_eval(poly, x);
    }
    
    double x_normalized = space_mapping_map(poly->space_transform,x);

    //values
    double p[2];
    double pnew;

    //gradients
    double pg[2];
    double pgnew;
        
    size_t iter = 0;
    double out = 0.0;
    p[0] = poly->p->const_term;
    pg[0] = 0.0;
    out += pg[0] * poly->coeff[iter];
    iter++;
    if (poly->num_poly > 1){
        p[1] = poly->p->lin_const + poly->p->lin_coeff * x_normalized;
        pg[1] = poly->p->lin_coeff;
        out += pg[1] * poly->coeff[iter];
        iter++;
    }

    double a,b,c;
    for (iter = 2; iter < poly->num_poly; iter++){
        a = poly->p->an(iter);
        b = poly->p->bn(iter);
        c = poly->p->cn(iter);
        pnew = eval_orth_poly_wp(poly->p, p[0], p[1], iter, x_normalized);
        
        pgnew = (a*x_normalized + b) * pg[1] + a * p[1] + c*pg[0];
        out += poly->coeff[iter] * pgnew;
        
        p[0] = p[1];
        p[1] = pnew;

        pg[0] = pg[1];
        pg[1] = pgnew;
    }

    out *= space_mapping_map_deriv(poly->space_transform,x);
    return out;
}


static inline double orth_poly_expansion_deriv_eval_for_approx(double x, void* poly){
    return orth_poly_expansion_deriv_eval(poly, x);
}

/********************************************************//**
*   Evaluate the derivative of an orthogonal polynomial expansion
*
*   \param[in] poly - pointer to orth poly expansion
*   \param[in] x    - location at which to evaluate
*
*
*   \return out - value of derivative
*************************************************************/
double cheb_expansion_deriv_eval(const struct OrthPolyExpansion * poly, double x)
{
    assert (poly != NULL);
    assert (poly->kristoffel_eval == 0);

    double dmult = space_mapping_map_deriv(poly->space_transform,x);
    if (poly->num_poly == 1){
        return 0.0;
    }
    else if (poly->num_poly == 2){
        return poly->coeff[1] * dmult;
    }

    double x_norm = space_mapping_map(poly->space_transform,x);
    
    if (poly->num_poly == 3){
        return (poly->coeff[1] + poly->coeff[2] * 4 * x_norm) * dmult;
    }

    double * cheb_eval = calloc_double(poly->num_poly);
    double * cheb_evald = calloc_double(poly->num_poly);
    cheb_eval[0] = 1.0;
    cheb_eval[1] = x_norm;
    cheb_eval[2] = 2.0*x_norm*cheb_eval[1] - cheb_eval[0];
    
    cheb_evald[0] = 0.0;
    cheb_evald[1] = 1.0;
    cheb_evald[2] = 4.0*x_norm;

    double out = poly->coeff[1]*cheb_evald[1] + poly->coeff[2]*cheb_evald[2];
    for (size_t ii = 3; ii < poly->num_poly; ii++){
        cheb_eval[ii] = 2.0 * x_norm * cheb_eval[ii-1] - cheb_eval[ii-2];
        cheb_evald[ii] = 2.0 * cheb_eval[ii-1] + 2.0 * x_norm * cheb_evald[ii-1] - 
            cheb_evald[ii-2];
        out += poly->coeff[ii]*cheb_evald[ii];
    }

    out *= dmult;
    free(cheb_eval); cheb_eval = NULL;
    free(cheb_evald); cheb_evald = NULL;
    return out;
}

static inline double cheb_expansion_deriv_eval_for_approx(double x, void* poly){
    return cheb_expansion_deriv_eval(poly, x);
}

/********************************************************//**
*   Evaluate the derivative of an orth poly expansion
*
*   \param[in] pin - orthogonal polynomial expansion
*   
*   \return derivative
*
*   \note
*       Could speed this up slightly by using partial sum
*       to keep track of sum of coefficients
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_deriv(const struct OrthPolyExpansion * pin)
{
    if (pin == NULL) return NULL;
    assert (pin->kristoffel_eval == 0);

    if (pin->p->ptype == FOURIER){
        return fourier_expansion_deriv(pin);
    }

    struct OrthPolyExpansion * p = orth_poly_expansion_copy(pin);
    orth_poly_expansion_round(&p);
            
    struct OrthPolyExpansion * out = NULL;

    out = orth_poly_expansion_copy(p);
    for (size_t ii = 0; ii < out->nalloc; ii++){
        out->coeff[ii] = 0.0;
    }
    if (p->num_poly == 1){
        orth_poly_expansion_round(&out);
        orth_poly_expansion_free(p); p = NULL;
        return out;
    }

    out->num_poly -= 1;
    if (p->p->ptype == LEGENDRE){
    /* if (1 == 0){ */

        double dtransform_dx = space_mapping_map_deriv(p->space_transform,0.0);
        for (size_t ii = 0; ii < p->num_poly-1; ii++){ // loop over coefficients
            for (size_t jj = ii+1; jj < p->num_poly; jj+=2){
                /* out->coeff[ii] += p->coeff[jj]; */
                out->coeff[ii] += p->coeff[jj]*sqrt(2*jj+1);
            }
            /* out->coeff[ii] *= (double) ( 2 * (ii) + 1) * dtransform_dx; */
            out->coeff[ii] *= sqrt((double) ( 2 * (ii) + 1))* dtransform_dx;
        }
    }
    else if (p->p->ptype == CHEBYSHEV){
        orth_poly_expansion_approx(cheb_expansion_deriv_eval_for_approx, p, out);      
    }
    else{
        orth_poly_expansion_approx(orth_poly_expansion_deriv_eval_for_approx, p, out);      
    }

    orth_poly_expansion_round(&out);
    orth_poly_expansion_free(p); p = NULL;
    return out;
}

/********************************************************//**
*   Evaluate the second derivative of a chebyshev expansion
*
*   \param[in] poly - pointer to orth poly expansion
*   \param[in] x    - location at which to evaluate
*
*
*   \return out - value of derivative
*************************************************************/
double cheb_expansion_dderiv_eval(const struct OrthPolyExpansion * poly, double x)
{
    assert (poly != NULL);
    assert (poly->kristoffel_eval == 0);

    
    double dmult = space_mapping_map_deriv(poly->space_transform,x);
    if (poly->num_poly <= 2){
        return 0.0;
    }
    else if (poly->num_poly == 3){
        return poly->coeff[2] * 4.0;
    }

    double x_norm = space_mapping_map(poly->space_transform,x);
    /* printf("x_norm = %3.15G\n", x_norm); */
    
    if (poly->num_poly == 4){
        /* printf("here yo!\n"); */
        return (poly->coeff[2] * 4 + poly->coeff[3]*24.0*x_norm) * dmult * dmult;
    }

    double * cheb_eval   = calloc_double(poly->num_poly);
    double * cheb_evald  = calloc_double(poly->num_poly);
    double * cheb_evaldd = calloc_double(poly->num_poly);
    cheb_eval[0] = 1.0;
    cheb_eval[1] = x_norm;
    cheb_eval[2] = 2.0*x_norm*cheb_eval[1] - cheb_eval[0];
    cheb_eval[3] = 2.0*x_norm*cheb_eval[2] - cheb_eval[1];
    
    cheb_evald[0] = 0.0;
    cheb_evald[1] = 1.0;
    cheb_evald[2] = 4.0*x_norm;
    cheb_evald[3] = 2.0 * cheb_eval[2] + 2.0 * x_norm * cheb_evald[2] - cheb_evald[1];

    cheb_evaldd[0] = 0.0;
    cheb_evaldd[1] = 0.0;
    cheb_evaldd[2] = 4.0;
    cheb_evaldd[3] = 24.0 * x_norm;

    double out = poly->coeff[2]*cheb_evaldd[2] + poly->coeff[3]*cheb_evaldd[3];
    for (size_t ii = 4; ii < poly->num_poly; ii++){
        cheb_eval[ii] = 2.0 * x_norm * cheb_eval[ii-1] - cheb_eval[ii-2];
        cheb_evald[ii] = 2.0 * cheb_eval[ii-1] + 2.0 * x_norm * cheb_evald[ii-1] - 
            cheb_evald[ii-2];
        cheb_evaldd[ii] = 4.0 * cheb_evald[ii-1] + 2.0 * x_norm * cheb_evaldd[ii-1] -
            cheb_evaldd[ii-2];

        out += poly->coeff[ii]*cheb_evaldd[ii];
    }

    out *= dmult*dmult;
    /* if (fabs(x_norm) > 0.999){ */
    /*     printf("out = %3.15G\n", out); */
    /* } */
    free(cheb_eval); cheb_eval = NULL;
    free(cheb_evald); cheb_evald = NULL;
    free(cheb_evaldd); cheb_evaldd = NULL;
    return out;
}

static inline double cheb_expansion_dderiv_eval_for_approx(double x, void* poly){
    return cheb_expansion_dderiv_eval(poly, x);
}
/********************************************************//**
*   Evaluate the second derivative of an orth poly expansion
*
*   \param[in] pin - orthogonal polynomial expansion
*   
*   \return derivative
*
*   \note
*       Could speed this up slightly by using partial sum
*       to keep track of sum of coefficients
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_dderiv(const struct OrthPolyExpansion * pin)
{
    if (pin == NULL) return NULL;
    assert (pin->kristoffel_eval == 0);

    if (pin->p->ptype == FOURIER){
        return fourier_expansion_dderiv(pin);
    }

    struct OrthPolyExpansion * p = orth_poly_expansion_copy(pin);
    orth_poly_expansion_round(&p);
            
    struct OrthPolyExpansion * out = NULL;

    out = orth_poly_expansion_copy(p);
    for (size_t ii = 0; ii < out->nalloc; ii++){
        out->coeff[ii] = 0.0;
    }
    if (p->num_poly < 2){
        orth_poly_expansion_free(p); p = NULL;
        return out;
    }

    /* printf("lets go!\n"); */
    /* dprint(p->num_poly, p->coeff); */
    out->num_poly -= 2;
    if (p->p->ptype == CHEBYSHEV){
        orth_poly_expansion_approx(cheb_expansion_dderiv_eval_for_approx, p, out);              
    }
    else{
        
        struct OrthPolyExpansion * temp = orth_poly_expansion_deriv(p);
        orth_poly_expansion_free(out);
        out = orth_poly_expansion_deriv(temp);
        orth_poly_expansion_free(temp); temp = NULL;
        /* fprintf(stderr, "Cannot yet take second derivative for polynomial of type %d\n", */
        /*         p->p->ptype); */
        /* exit(1); */
    }

    orth_poly_expansion_round(&out);
    orth_poly_expansion_free(p); p = NULL;
    return out;
}

/********************************************************//**
   Take a second derivative and enforce periodic bc
**************************************************************/
struct OrthPolyExpansion * orth_poly_expansion_dderiv_periodic(const struct OrthPolyExpansion * f)
{
    if (f->p->ptype == FOURIER){
        return orth_poly_expansion_dderiv(f);
    }
    else{
        NOT_IMPLEMENTED_MSG("orth_poly_expansion_dderiv_periodic");
        exit(1);
    }
}

/********************************************************//**
*   free the memory of an orthonormal polynomial expansion
*
*   \param[in,out] p - orthogonal polynomial expansion
*************************************************************/
void orth_poly_expansion_free(struct OrthPolyExpansion * p){
    if (p != NULL){
        if (p->p->ptype == FOURIER){
            free(p->ccoeff); p->ccoeff = NULL;
        }
        free_orth_poly(p->p); p->p = NULL;
        space_mapping_free(p->space_transform); p->space_transform = NULL;
        free(p->coeff); p->coeff = NULL;

        free(p); p = NULL;
    }
}

/********************************************************//**
*   Serialize orth_poly_expansion
*   
*   \param[in] ser       - location to which to serialize
*   \param[in] p         - polynomial
*   \param[in] totSizeIn - if not null then only return total size of 
*                          array without serialization! if NULL then serialiaze
*
*   \return ptr : pointer to end of serialization
*************************************************************/
unsigned char *
serialize_orth_poly_expansion(unsigned char * ser, 
        struct OrthPolyExpansion * p,
        size_t * totSizeIn)
{
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff

    size_t totsize;
    if (p->p->ptype != FOURIER){
        totsize = sizeof(int) + 2*sizeof(double) + 
            p->num_poly * sizeof(double) + sizeof(size_t);
    }
    else{
        /* fprintf(stderr, "Cannot serialized fourier polynomials yet\n"); */
        /* exit(1); */
        totsize = sizeof(int) + 2*sizeof(double) + 
            2*p->num_poly * sizeof(double) + sizeof(size_t);
    }

    size_t size_mapping;
    serialize_space_mapping(NULL,p->space_transform,&size_mapping);
    totsize += size_mapping;

    totsize += sizeof(int); // for kristoffel flag
    if (totSizeIn != NULL){
        *totSizeIn = totsize;
        return ser;
    }
    unsigned char * ptr = serialize_int(ser, p->p->ptype);
    ptr = serialize_double(ptr, p->lower_bound);
    ptr = serialize_double(ptr, p->upper_bound);
    if (p->p->ptype != FOURIER){
        ptr = serialize_doublep(ptr, p->coeff, p->num_poly);
    }
    else{
        ptr = serialize_size_t(ptr, p->num_poly);
        for (size_t ii = 0; ii < p->num_poly; ii++){
            ptr = serialize_double(ptr, creal(p->ccoeff[ii]));
            ptr = serialize_double(ptr, cimag(p->ccoeff[ii]));
        }
    }
    
    ptr = serialize_space_mapping(ptr,p->space_transform,NULL);
    ptr = serialize_int(ptr,p->kristoffel_eval);
    return ptr;
}

/********************************************************//**
*   Deserialize orth_poly_expansion
*
*   \param[in]     ser  - input string
*   \param[in,out] poly - poly expansion
*
*   \return ptr - ser + number of bytes of poly expansion
*************************************************************/
unsigned char * 
deserialize_orth_poly_expansion(
    unsigned char * ser, 
    struct OrthPolyExpansion ** poly)
{
    
    size_t num_poly = 0;
    //size_t npoly_check = 0;
    double lower_bound = 0;
    double upper_bound = 0;
    double * coeff = NULL;
    complex double * ccoeff = NULL;
    struct OrthPoly * p = NULL;
    struct SpaceMapping * map = NULL;
    // order is  ptype->lower_bound->upper_bound->orth_poly->coeff
    p = deserialize_orth_poly(ser);
    unsigned char * ptr = ser + sizeof(int);
    ptr = deserialize_double(ptr,&lower_bound);
    ptr = deserialize_double(ptr,&upper_bound);

    if (p->ptype != FOURIER){
        ptr = deserialize_doublep(ptr, &coeff, &num_poly);
    }
    else{
        ptr = deserialize_size_t(ptr, &num_poly);
        /* printf("num_poly = %zu\n", num_poly); */
        ccoeff = malloc( num_poly * sizeof(complex double));
        for (size_t ii = 0; ii < num_poly; ii++){
            double cr;
            double ci;
            ptr = deserialize_double(ptr, &cr);
            ptr = deserialize_double(ptr, &ci);
            ccoeff[ii] = cr + ci * (complex double) I;
            /* printf("%3.5f, %3.5f\n", cr, ci); */
        }
        
    }
    ptr = deserialize_space_mapping(ptr, &map);

    int kristoffel_eval;
    ptr = deserialize_int(ptr,&kristoffel_eval);
    if ( NULL == (*poly = malloc(sizeof(struct OrthPolyExpansion)))){
        fprintf(stderr, "failed to allocate memory for poly exp.\n");
        exit(1);
    }
    (*poly)->num_poly = num_poly;
    (*poly)->lower_bound = lower_bound;
    (*poly)->upper_bound = upper_bound;
    (*poly)->coeff = coeff;
    (*poly)->ccoeff = ccoeff;
    (*poly)->nalloc = num_poly;//+OPECALLOC;
    (*poly)->p = p;
    (*poly)->space_transform = map;
    (*poly)->kristoffel_eval = kristoffel_eval;
    return ptr;
}

/********************************************************//**
    Save an orthonormal polynomial expansion in text format

    \param[in] f      - function to save
    \param[in] stream - stream to save it to
    \param[in] prec   - precision with which to save it
************************************************************/
void orth_poly_expansion_savetxt(const struct OrthPolyExpansion * f,
                                 FILE * stream, size_t prec)
{
    assert (f != NULL);
    if (f->p->ptype == FOURIER){
        fprintf(stderr, "Cannot savetxt fourier polynomials yet\n");
        exit(1);
    }
    
    fprintf(stream,"%d ",f->p->ptype);
    fprintf(stream,"%3.*G ",(int)prec,f->lower_bound);
    fprintf(stream,"%3.*G ",(int)prec,f->upper_bound);
    fprintf(stream,"%zu ",f->num_poly);
    for (size_t ii = 0; ii < f->num_poly; ii++){
        if (prec < 100){
            fprintf(stream, "%3.*G ",(int)prec,f->coeff[ii]);
        }
    }
    space_mapping_savetxt(f->space_transform,stream,prec);
    fprintf(stream,"%d ",f->kristoffel_eval);
}

/********************************************************//**
    Load an orthonormal polynomial expansion in text format

    \param[in] stream - stream to save it to

    \return Orthonormal polynomial expansion
************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_loadtxt(FILE * stream)//l, size_t prec)
{

    enum poly_type ptype;
    double lower_bound = 0;
    double upper_bound = 0;
    size_t num_poly;

    int ptypeint;
    int num = fscanf(stream,"%d ",&ptypeint);
    ptype = (enum poly_type)ptypeint;
    assert (num == 1);
    num = fscanf(stream,"%lG ",&lower_bound);
    assert (num == 1);
    num = fscanf(stream,"%lG ",&upper_bound);
    assert (num == 1);
    num = fscanf(stream,"%zu ",&num_poly);
    assert (num == 1);

    struct OrthPolyExpansion * ope = 
        orth_poly_expansion_init(ptype,num_poly,lower_bound,upper_bound);

    for (size_t ii = 0; ii < ope->num_poly; ii++){
        num = fscanf(stream, "%lG ",ope->coeff+ii);
        assert (num == 1);
    }

    space_mapping_free(ope->space_transform); ope->space_transform = NULL;
    ope->space_transform = space_mapping_loadtxt(stream);

    int kristoffel_eval;
    num = fscanf(stream,"%d ",&kristoffel_eval);
    assert (num == 1);
    ope->kristoffel_eval = kristoffel_eval;
    
    return ope;
}

/********************************************************//**
*   Convert an orthogonal polynomial expansion to a standard_polynomial
*
*   \param[in] p - polynomial
*
*   \return sp - standard polynomial
*************************************************************/
struct StandardPoly * 
orth_poly_expansion_to_standard_poly(struct OrthPolyExpansion * p)
{
    if (p->p->ptype == FOURIER){
        fprintf(stderr, "Cannot convert fourier poly to a standard poly");
        exit(1);
    }
    struct StandardPoly * sp = 
        standard_poly_init(p->num_poly,p->lower_bound,p->upper_bound);
    
    double m = (p->p->upper - p->p->lower) / (p->upper_bound - p->lower_bound);
    double off = p->p->upper - m * p->upper_bound;

    size_t ii, jj;
    size_t n = p->num_poly-1;

    sp->coeff[0] = p->coeff[0]*p->p->const_term;

    if (n > 0){
        sp->coeff[0]+=p->coeff[1] * (p->p->lin_const + p->p->lin_coeff * off);
        sp->coeff[1]+=p->coeff[1] * p->p->lin_coeff * m;
    }
    if (n > 1){
        
        double * a = calloc_double(n+1); //n-2 poly
        a[0] = p->p->const_term;
        double * b = calloc_double(n+1); // n- 1poly
        double * c = calloc_double(n+1); // n- 1poly
        b[0] = p->p->lin_const + p->p->lin_coeff * off;
        b[1] = p->p->lin_coeff * m;
        for (ii = 2; ii < n+1; ii++){ // starting at the order 2 polynomial
            c[0] = (p->p->bn(ii) + p->p->an(ii)*off) * b[0] + 
                                                        p->p->cn(ii) * a[0];
            sp->coeff[0] += p->coeff[ii] * c[0];
            for (jj = 1; jj < ii-1; jj++){
                c[jj] = (p->p->an(ii) * m) * b[jj-1] + 
                        (p->p->bn(ii) + p->p->an(ii) * off) * b[jj] + 
                        p->p->cn(ii) * a[jj];
                sp->coeff[jj] += p->coeff[ii] * c[jj];
            }
            c[ii-1] = (p->p->an(ii) * m) * b[ii-2] + 
                            (p->p->bn(ii) + p->p->an(ii) * off) * b[ii-1];
            c[ii] = (p->p->an(ii) * m) * b[ii-1];
            
            sp->coeff[ii-1] += p->coeff[ii] * c[ii-1];
            sp->coeff[ii] += p->coeff[ii] * c[ii];

            memcpy(a, b, ii * sizeof(double));
            memcpy(b, c, (ii+1) * sizeof(double));
        }
        
        free(a);
        free(b);
        free(c);
    }

    // Need to do something with lower and upper bounds!!
    return sp;
}

/********************************************************//**
*   Evaluate each orthonormal polynomial expansion that is in an 
*   array of generic functions 
*
*   \param[in]     n       - number of polynomials
*   \param[in]     parr    - polynomial expansions
*   \param[in]     x       - location at which to evaluate
*   \param[in,out] out     - evaluations
*
*   \return 0 - successful
*
*   \note
*   Assumes all functions have the same bounds
*************************************************************/
int orth_poly_expansion_arr_eval(size_t n,
                                 struct OrthPolyExpansion ** parr, 
                                 double x, double * out)
{


    if (parr[0]->kristoffel_eval == 1){
        for (size_t ii = 0; ii < n; ii++){
            out[ii] = orth_poly_expansion_eval(parr[ii],x);
        }
        return 0;
    }

    int all_same = 1;
    enum poly_type ptype = parr[0]->p->ptype;
    for (size_t ii = 1; ii < n; ii++){
        if (parr[ii]->p->ptype != ptype){
            all_same = 0;
            break;
        }
    }

    if ((all_same == 0) || (ptype == CHEBYSHEV) || (ptype == FOURIER)){
        for (size_t ii = 0; ii < n; ii++){
            out[ii] = orth_poly_expansion_eval(parr[ii],x);
        }
        return 0;
    }

    // all the polynomials are of the same type
    size_t maxpoly = 0;
    for (size_t ii = 0; ii < n; ii++){
        if (parr[ii]->num_poly > maxpoly){
            maxpoly = parr[ii]->num_poly;
        }
        out[ii] = 0.0;
    }


    double x_norm = space_mapping_map(parr[0]->space_transform,x);

    // double out = 0.0;
    double p[2];
    double pnew;
    p[0] = parr[0]->p->const_term;
    size_t iter = 0;
    for (size_t ii = 0; ii < n; ii++){
        out[ii] += p[0] * parr[ii]->coeff[iter];
    }
    iter++;
    p[1] = parr[0]->p->lin_const + parr[0]->p->lin_coeff * x_norm;
    for (size_t ii = 0; ii < n; ii++){
        if (parr[ii]->num_poly > iter){
            out[ii] += p[1] * parr[ii]->coeff[iter];
        }
    }
    iter++;
    for (iter = 2; iter < maxpoly; iter++){
        pnew = eval_orth_poly_wp(parr[0]->p, p[0], p[1], iter, x_norm);
        for (size_t ii = 0; ii < n; ii++){
            if (parr[ii]->num_poly > iter){
                out[ii] += parr[ii]->coeff[iter] * pnew;
            }
        }
        p[0] = p[1];
        p[1] = pnew;
    }

    return 0;
}

/********************************************************//**
*   Evaluate each orthonormal polynomial expansion that is in an 
*   array of generic functions at an array of points
*
*   \param[in]     n          - number of polynomials
*   \param[in]     parr       - polynomial expansions (all have the same bounds)
*   \param[in]     N          - number of evaluations
*   \param[in]     x          - locations at which to evaluate
*   \param[in]     incx       - increment between locations
*   \param[in,out] y          - evaluations
*   \param[in]     incy       - increment between evaluations of array (at least n)
*
*   \return 0 - successful
*
*   \note
*   Assumes all functions have the same bounds
*************************************************************/
int orth_poly_expansion_arr_evalN(size_t n,
                                  struct OrthPolyExpansion ** parr,
                                  size_t N,
                                  const double * x, size_t incx,
                                  double * y, size_t incy)
{
    if (parr[0]->kristoffel_eval == 1){
        for (size_t jj = 0; jj < N; jj++){
            for (size_t ii = 0; ii < n; ii++){
                y[ii + jj * incy] = orth_poly_expansion_eval(parr[ii],x[jj*incx]);
                /* printf("y = %G\n",y[ii+jj*incy]); */
            }
        }
        return 0;
    }
    

    for (size_t jj = 0; jj < N; jj++){
        for (size_t ii = 0; ii < n; ii++){
            y[ii + jj * incy] = 0.0;
        }
    }

    int res;
    for (size_t jj = 0; jj < N; jj++){
        res = orth_poly_expansion_arr_eval(n, parr, 
                                           x[jj*incx], y + jj*incy);
        if (res != 0){
            return res;
        }
    }


    for (size_t jj = 0; jj < N; jj++){
        for (size_t ii = 0; ii < n; ii++){
            if (isnan(y[ii + jj* incy]) || y[ii+jj * incy] > 1e100){
                fprintf(stderr,"Warning, evaluation in orth_poly_expansion_array_eval is nan\n");
                fprintf(stderr,"Polynomial %zu, evaluation %zu\n",ii,jj);
                print_orth_poly_expansion(parr[ii],0,NULL,stderr);
                exit(1);
            }
            else if (isinf(y[ii + jj * incy])){
                fprintf(stderr,"Warning, evaluation in orth_poly_expansion_array_eval inf\n");
                exit(1);
            }        
        }
    }
    
    
    return 0;
}

/********************************************************//**
*   Evaluate a Chebyshev polynomial expansion using clenshaw algorithm
*
*   \param[in] poly - polynomial expansion
*   \param[in] x    - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double chebyshev_poly_expansion_eval(const struct OrthPolyExpansion * poly, double x)
{

    assert (poly->kristoffel_eval == 0);

    double p[2] = {0.0, 0.0};
    double pnew;
    
    double x_norm = space_mapping_map(poly->space_transform,x);
    size_t n = poly->num_poly-1;
    while (n > 0){
        pnew = poly->coeff[n] + 2.0 * x_norm * p[0] - p[1];
        p[1] = p[0];
        p[0] = pnew;
        n--;
    }
    double out = poly->coeff[0] + x_norm * p[0] - p[1];
    return out;
}

/********************************************************//**
*   Evaluate a polynomial expansion consisting of sequentially increasing 
*   order polynomials from the same family.
*
*   \param[in] poly - polynomial expansion
*   \param[in] x    - location at which to evaluate
*
*   \return out - polynomial value
*************************************************************/
double orth_poly_expansion_eval(const struct OrthPolyExpansion * poly, double x)
{
    double out = 0.0;
    if (poly->p->ptype == FOURIER){
        out = fourier_expansion_eval(poly,x);
    }
    else if (poly->p->ptype != CHEBYSHEV){
        size_t iter = 0;
        double p [2];
        double pnew;
        
        double x_normalized = space_mapping_map(poly->space_transform,x);

        double den = 0.0;
        
        p[0] = poly->p->const_term;
        out += p[0] * poly->coeff[iter];

        den += p[0]*p[0];
        
        iter++;
        if (poly->num_poly > 1){
            p[1] = poly->p->lin_const + poly->p->lin_coeff * x_normalized;
            out += p[1] * poly->coeff[iter];

            den += p[1]*p[1];
            iter++;
        }
        for (iter = 2; iter < poly->num_poly; iter++){
            pnew = eval_orth_poly_wp(poly->p, p[0], p[1], iter, x_normalized);
            out += poly->coeff[iter] * pnew;
            p[0] = p[1];
            p[1] = pnew;

            den += pnew*pnew;
        }


        if (poly->kristoffel_eval == 1){
            /* printf("normalizing for kristoffel out = %G, %G\n",out,den); */
            /* dprint(poly->num_poly,poly->coeff); */
            /* out /= sqrt(den); */
            out /= sqrt( den / ( (double) poly->num_poly )) ;
        }
    }
    else{
        out = chebyshev_poly_expansion_eval(poly,x);
    }
    return out;
}

/********************************************************//**
*   Get the kristoffel weight of an orthonormal polynomial expansion
*
*   \param[in] poly - polynomial expansion
*   \param[in] x    - location at which to evaluate
*
*   \return out - weight
*************************************************************/
double orth_poly_expansion_get_kristoffel_weight(const struct OrthPolyExpansion * poly, double x)
{
    assert (poly != NULL);
    if (poly->p->ptype == FOURIER){
        fprintf(stderr,
                "Cannot get kristoffel_weight for fourier basis\n");
        exit(1);
    }
    size_t iter = 0;
    double p [2];
    double pnew;
        
    double x_normalized = space_mapping_map(poly->space_transform,x);
    double den = 0.0;
        
    p[0] = poly->p->const_term;


    den += p[0]*p[0];
        
    iter++;
    if (poly->num_poly > 1){
        p[1] = poly->p->lin_const + poly->p->lin_coeff * x_normalized;

        den += p[1]*p[1];
        iter++;
    }
    for (iter = 2; iter < poly->num_poly; iter++){
        pnew = eval_orth_poly_wp(poly->p, p[0], p[1], iter, x_normalized);

        p[0] = p[1];
        p[1] = pnew;

        den += pnew*pnew;
    }

    // Normalize by number of functions
    return sqrt(den / ( (double) poly->num_poly ) );
    //return sqrt(den);
}

/********************************************************//**
*   Evaluate a polynomial expansion consisting of sequentially increasing 
*   order polynomials from the same family.
*
*   \param[in]     poly - function
*   \param[in]     N    - number of evaluations
*   \param[in]     x    - location at which to evaluate
*   \param[in]     incx - increment of x
*   \param[in,out] y    - allocated space for evaluations
*   \param[in]     incy - increment of y*
*
*   \note Currently just calls the single evaluation code
*         Note sure if this is optimal, cache-wise
*************************************************************/
void orth_poly_expansion_evalN(const struct OrthPolyExpansion * poly, size_t N,
                               const double * x, size_t incx, double * y, size_t incy)
{
    for (size_t ii = 0; ii < N; ii++){
        y[ii*incy] = orth_poly_expansion_eval(poly,x[ii*incx]);
    }
}

/********************************************************//*
*   Evaluate the gradient of an orthonormal polynomial expansion 
*   with respect to the parameters
*
*   \param[in]     poly - polynomial expansion
*   \param[in]     nx   - number of x points
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return 0 success, 1 otherwise
*************************************************************/
int orth_poly_expansion_param_grad_eval(
    const struct OrthPolyExpansion * poly, size_t nx, const double * x, double * grad)
{
    if (poly->p->ptype == FOURIER){
        fprintf(stderr, "Cannot perform param_grad_eval with fourier basis\n");
        exit(1);
    }
    size_t nparam = orth_poly_expansion_get_num_params(poly);
    for (size_t ii = 0; ii < nx; ii++){

        double p[2];
        double pnew;

        double x_norm = space_mapping_map(poly->space_transform,x[ii]);
    
        size_t iter = 0;
        p[0] = poly->p->const_term;
        double den = p[0]*p[0];
        
        grad[ii*nparam] = p[0];
        iter++;
        if (poly->num_poly > 1){
            p[1] = poly->p->lin_const + poly->p->lin_coeff * x_norm;
            grad[ii*nparam + iter] = p[1]; 
            iter++;
            den += p[1]*p[1];

            for (iter = 2; iter < poly->num_poly; iter++){
                pnew = (poly->p->an(iter)*x_norm + poly->p->bn(iter)) * p[1] + poly->p->cn(iter) * p[0];
                grad[ii*nparam + iter] = pnew;
                den += pnew * pnew;
                p[0] = p[1];
                p[1] = pnew;
            }
        }

        if (poly->kristoffel_eval == 1){
            /* printf("gradient normalized by kristoffel %G\n",den); */
            for (size_t jj = 0; jj < poly->num_poly; jj++){
                // Normalize by number of functions
                grad[ii*nparam+jj] /= sqrt(den / ( (double) poly->num_poly ));
                //grad[ii*nparam+jj] /= sqrt(den);
            }
        }
    }
    return 0;    
}


/********************************************************//*
*   Evaluate the gradient of an orthonormal polynomial expansion 
*   with respect to the parameters
*
*   \param[in]     poly - polynomial expansion
*   \param[in]     x    - location at which to evaluate
*   \param[in,out] grad - gradient values (N,nx)
*
*   \return evaluation
*************************************************************/
double orth_poly_expansion_param_grad_eval2(
    const struct OrthPolyExpansion * poly, double x, double * grad)
{
    if (poly->p->ptype == FOURIER){
        fprintf(stderr, "Cannot perform param_grad_eval with fourier basis\n");
        exit(1);
    }
    
    double out = 0.0;

    double p[2];
    double pnew;

    double x_norm = space_mapping_map(poly->space_transform,x);

    double den = 0.0;
    size_t iter = 0;
    p[0] = poly->p->const_term;
    grad[0] = p[0];
    den += p[0]*p[0];
    
    out += p[0]*poly->coeff[0];
    iter++;
    if (poly->num_poly > 1){
        p[1] = poly->p->lin_const + poly->p->lin_coeff * x_norm;
        grad[iter] = p[1];
        den += p[1]*p[1];
        out += p[1]*poly->coeff[1];
        iter++;

        for (iter = 2; iter < poly->num_poly; iter++){
            pnew = (poly->p->an(iter)*x_norm + poly->p->bn(iter)) * p[1] + poly->p->cn(iter) * p[0];
            grad[iter] = pnew;
            out += pnew*poly->coeff[iter];
            p[0] = p[1];
            p[1] = pnew;

            den += pnew * pnew;
        }
    }
    if (poly->kristoffel_eval == 1){
        /* printf("gradient normalized by kristoffel %G\n",den); */
        for (size_t jj = 0; jj < poly->num_poly; jj++){
            // Normalize by number of functions
            grad[jj] /= sqrt(den / ( (double) poly->num_poly ));
            /* grad[jj] /= sqrt(den); */
        }
    }
    return out;    
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     poly  - polynomial
    \param[in]     scale - scaling for additional gradient
    \param[in,out] grad  - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

************************************************************/
int
orth_poly_expansion_squared_norm_param_grad(const struct OrthPolyExpansion * poly,
                                            double scale, double * grad)
{
    if (poly->p->ptype == FOURIER){
        fprintf(stderr, "Cannot perform squared_norm_param_grad with fourier basis\n");
        exit(1);
    }
    
    assert (poly->kristoffel_eval == 0);
    int res = 1;

    
    // assuming linear transformation
    double dtransform_dx = space_mapping_map_deriv(poly->space_transform,0.0);
    if (poly->p->ptype == LEGENDRE){
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            //the extra 2 is for the weight
            grad[ii] += 2.0*scale * poly->coeff[ii] * poly->p->norm(ii) * /* *dtransform_dx * */
                         (poly->upper_bound-poly->lower_bound); 
        }
        res = 0;
    }
    else if (poly->p->ptype == HERMITE){
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            grad[ii] += 2.0*scale * poly->coeff[ii] * poly->p->norm(ii) * dtransform_dx;
        }
        res = 0;
    }
    else if (poly->p->ptype == CHEBYSHEV){
        double * temp = calloc_double(poly->num_poly);
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            temp[ii] = 2.0/(1.0 - (double)2*ii*2*ii);
        }
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            for (size_t jj = 0; jj < poly->num_poly; jj++){
                size_t n1 = ii+jj;
                size_t n2;
                if (ii > jj){
                    n2 = ii-jj;
                }
                else{
                    n2 = jj-ii;
                }
                if (n1 % 2 == 0){
                    grad[ii] += 2.0*scale*temp[n1/2] * dtransform_dx;
                }
                if (n2 % 2 == 0){
                    grad[ii] += 2.0*scale*temp[n2/2] * dtransform_dx;
                }
            }
        }
        free(temp); temp = NULL;
        res = 0;
    }
    else{
        fprintf(stderr,
                "Cannot evaluate derivative with respect to parameters for polynomial type %d\n",
                poly->p->ptype);
        exit(1);
    }
    return res;
}

/********************************************************//**
    Squared norm of a function in RKHS 

    \param[in]     poly        - polynomial
    \param[in]     decay_type  - type of decay
    \param[in]     decay_param - parameter of decay

    \return  0 - success, 1 -failure
************************************************************/
double
orth_poly_expansion_rkhs_squared_norm(const struct OrthPolyExpansion * poly,
                                      enum coeff_decay_type decay_type,
                                      double decay_param)
{

    assert (poly->kristoffel_eval == 0);
    if (poly->p->ptype == FOURIER){
        fprintf(stderr, "Cannot perform rkhs_squared_norm with fourier basis\n");
        exit(1);
    }
    
    // assuming linear transformation
    double m = space_mapping_map_deriv(poly->space_transform,0.0);
    /* double m = (poly->upper_bound-poly->lower_bound) /(poly->p->upper - poly->p->lower); */
    double out = 0.0;
    if (poly->p->ptype == LEGENDRE){
        if (decay_type == ALGEBRAIC){
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                out += poly->coeff[ii] * poly->coeff[ii]*pow(decay_param,ii) * poly->p->norm(ii)*2.0 * m;
            }   
        }
        else if (decay_type == EXPONENTIAL){
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                out += poly->coeff[ii] * poly->coeff[ii]*pow((double)ii+1.0,-decay_param)*m*poly->p->norm(ii)*2.0;
            }   
        }
        else{
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                out += poly->coeff[ii] * poly->coeff[ii] * poly->p->norm(ii)*2.0*m;
            }
        }
    }
    else if (poly->p->ptype == HERMITE){
        if (decay_type == ALGEBRAIC){
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                out += poly->coeff[ii] * poly->coeff[ii]*pow(decay_param,ii) * poly->p->norm(ii);
            }   
        }
        else if (decay_type == EXPONENTIAL){
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                out += poly->coeff[ii] * poly->coeff[ii]*pow((double)ii+1.0,-decay_param) * poly->p->norm(ii);
            }   
        }
        else{
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                out += poly->coeff[ii] * poly->coeff[ii] * poly->p->norm(ii);
            }
        }
    }
    else if (poly->p->ptype == CHEBYSHEV){
        double * temp = calloc_double(poly->num_poly);
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            temp[ii] = 2.0/(1.0 - (double)2*ii*2*ii);
        }
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            double temp_sum = 0.0;
            for (size_t jj = 0; jj < poly->num_poly; jj++){
                size_t n1 = ii+jj;
                size_t n2;
                if (ii > jj){
                    n2 = ii-jj;
                }
                else{
                    n2 = jj-ii;
                }
                if (n1 % 2 == 0){
                    temp_sum += poly->coeff[jj]*temp[n1/2];
                }
                if (n2 % 2 == 0){
                    temp_sum += poly->coeff[jj]*temp[n2/2];
                }
            }
            if (decay_type == ALGEBRAIC){
                out += temp_sum*temp_sum * pow(decay_param,ii)*m;
            }
            else if (decay_type == EXPONENTIAL){
                out += temp_sum*temp_sum * pow((double)ii+1.0,-decay_param)*m;
            }
            else{
                out += temp_sum * temp_sum*m;
            }
        }
        free(temp); temp = NULL;
    }
    else{
        fprintf(stderr, "Cannot evaluate derivative with respect to parameters for polynomial type %d\n",poly->p->ptype);
        exit(1);
    }
    return out;
}

/********************************************************//**
    Take a gradient of the squared norm 
    with respect to its parameters, and add a scaled version
    of this gradient to *grad*

    \param[in]     poly        - polynomial
    \param[in]     scale       - scaling for additional gradient
    \param[in]     decay_type  - type of decay
    \param[in]     decay_param - parameter of decay
    \param[in,out] grad        - gradient, on output adds scale * new_grad

    \return  0 - success, 1 -failure

    \note 
    NEED TO DO SOME TESTS FOR CHEBYSHEV (dont use for now)
************************************************************/
int
orth_poly_expansion_rkhs_squared_norm_param_grad(const struct OrthPolyExpansion * poly,
                                                 double scale, enum coeff_decay_type decay_type,
                                                 double decay_param, double * grad)
{
    assert (poly->kristoffel_eval == 0);
    if (poly->p->ptype == FOURIER){
        fprintf(stderr, "Cannot perform rkhs_squared_norm with fourier basis\n");
        exit(1);
    }    
    int res = 1;
    if ((poly->p->ptype == LEGENDRE) || (poly->p->ptype == HERMITE)){
        if (decay_type == ALGEBRAIC){
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                grad[ii] += 2.0*scale * poly->coeff[ii] * pow(decay_param,ii);
            }   
        }
        else if (decay_type == EXPONENTIAL){
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                grad[ii] += 2.0*scale * poly->coeff[ii] * pow((double)ii+1.0,-decay_param);
            }   
        }
        else{
            for (size_t ii = 0; ii < poly->num_poly; ii++){
                grad[ii] += 2.0*scale * poly->coeff[ii];
            }
        }
        res = 0;
    }
    else if (poly->p->ptype == CHEBYSHEV){
        // THIS COULD BE WRONG!!
        double * temp = calloc_double(poly->num_poly);
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            temp[ii] = 2.0/(1.0 - (double)2*ii*2*ii);
        }
        for (size_t ii = 0; ii < poly->num_poly; ii++){
            for (size_t jj = 0; jj < poly->num_poly; jj++){
                size_t n1 = ii+jj;
                size_t n2;
                if (ii > jj){
                    n2 = ii-jj;
                }
                else{
                    n2 = jj-ii;
                }
                if (decay_type == ALGEBRAIC){
                    if (n1 % 2 == 0){
                        grad[ii] += 2.0*scale*temp[n1/2]* pow(decay_param,ii);
                    }
                    if (n2 % 2 == 0){
                        grad[ii] += 2.0*scale*temp[n2/2]* pow(decay_param,ii);
                    }
                }
                else if (decay_type == EXPONENTIAL){
                    if (n1 % 2 == 0){
                        grad[ii] += 2.0*scale*temp[n1/2]*pow((double)ii+1.0,-decay_param);
                    }
                    if (n2 % 2 == 0){
                        grad[ii] += 2.0*scale*temp[n2/2]*pow((double)ii+1.0,-decay_param);
                    }
                }
                else {
                    if (n1 % 2 == 0){
                        grad[ii] += 2.0*scale*temp[n1/2];
                    }
                    if (n2 % 2 == 0){
                        grad[ii] += 2.0*scale*temp[n2/2];
                    }
                }
            }
        }
        free(temp); temp = NULL;
        res = 0;
    }
    else{
        fprintf(stderr, "Cannot evaluate derivative with respect to parameters for polynomial type %d\n",poly->p->ptype);
        exit(1);
    }
    return res;
}

/********************************************************//**
*  Round an orthogonal polynomial expansion
*
*  \param[in,out] p - orthogonal polynomial expansion
*
*  \note
*      (UNTESTED, use with care!!!! 
*************************************************************/
void orth_poly_expansion_round(struct OrthPolyExpansion ** p)
{   
    if ((0 == 0) && ((*p)->p->ptype != FOURIER)){
        /* double thresh = 1e-3*ZEROTHRESH; */
        double thresh = ZEROTHRESH;
        /* double thresh = 1e-30; */
        /* double thresh = 10.0*DBL_EPSILON; */
        /* printf("thresh = %G\n",thresh); */
        size_t jj = 0;
        //
        int allzero = 1;
        double maxcoeff = fabs((*p)->coeff[0]);
        for (size_t ii = 1; ii < (*p)->num_poly; ii++){
            double val = fabs((*p)->coeff[ii]);
            if (val > maxcoeff){
                maxcoeff = val;
            }
        }
        maxcoeff = maxcoeff * (*p)->num_poly;
        /* printf("maxcoeff = %3.15G\n", maxcoeff); */
	    for (jj = 0; jj < (*p)->num_poly;jj++){
            if (fabs((*p)->coeff[jj]) < thresh){
                (*p)->coeff[jj] = 0.0;
            }
            if (fabs((*p)->coeff[jj])/maxcoeff < thresh){
                (*p)->coeff[jj] = 0.0;
            }
            else{
                allzero = 0;
            }
           
	    }
        if (allzero == 1){
            (*p)->num_poly = 1;
        }
        else {
            jj = 0;
            size_t end = (*p)->num_poly;
            if ((*p)->num_poly > 2){
                while (fabs((*p)->coeff[end-1]) < thresh){
                    end-=1;
                    if (end == 0){
                        break;
                    }
                }
                
                if (end > 0){
                    //printf("SHOULD NOT BE HERE\n");
                    size_t num_poly = end;
                    //
                    //double * new_coeff = calloc_double(num_poly);
                    //for (jj = 0; jj < num_poly; jj++){
                    //    new_coeff[jj] = (*p)->coeff[jj];
                   // }
                    //free((*p)->coeff); (*p)->coeff=NULL;
                    //(*p)->coeff = new_coeff;
                    (*p)->num_poly = num_poly;
                }
            }
        }

        /* printf("rounded coeffs = "); dprint((*p)->num_poly, (*p)->coeff); */

        /* orth_poly_expansion_roundt(p,thresh); */

    }
}

/********************************************************//**
*  Round an orthogonal polynomial expansion to a threshold
*
*  \param[in,out] p      - orthogonal polynomial expansion
*  \param[in]     thresh - threshold (relative) to round to
*
*  \note
*      (UNTESTED, use with care!!!! 
*************************************************************/
void orth_poly_expansion_roundt(struct OrthPolyExpansion ** p, double thresh)
{   
    
    size_t jj = 0;
    double sum = 0.0;
    /* double maxval = fabs((*p)->coeff[0]); */
	/* for (jj = 1; jj < (*p)->num_poly;jj++){ */
    /*     sum += pow((*p)->coeff[jj],2); */
    /*     if (fabs((*p)->coeff[jj]) > maxval){ */
    /*         maxval = fabs((*p)->coeff[jj]); */
    /*     } */
	/* } */
    size_t keep = (*p)->num_poly;
    if (sum <= thresh){
        keep = 1;
    }
    else{
        double sumrun = 0.0;
        for (jj = 0; jj < (*p)->num_poly; jj++){
            /* if ((fabs((*p)->coeff[jj]) / maxval) < thresh){ */
            /*     (*p)->coeff[jj] = 0.0; */
            /* } */
            sumrun += pow((*p)->coeff[jj],2);
            if ( (sumrun / sum) > (1.0-thresh)){
                keep = jj+1;
                break;
            }
        }
    }
    /* dprint((*p)->num_poly, (*p)->coeff); */
    /* printf("number keep = %zu\n",keep); */
    //printf("tolerance = %G\n",thresh);
    double * new_coeff = calloc_double(keep + OPECALLOC);
    memmove(new_coeff,(*p)->coeff, keep * sizeof(double));
    free((*p)->coeff);
    (*p)->num_poly = keep;
    (*p)->nalloc = (*p)->num_poly + OPECALLOC;
    (*p)->coeff = new_coeff;
}



/********************************************************//**
*  Approximate a function with an orthogonal polynomial
*  series with a fixed number of basis
*
*  \param[in] A    - function to approximate
*  \param[in] args - arguments to function
*  \param[in] poly - orthogonal polynomial expansion
*
*  \note
*       Wont work for polynomial expansion with only the constant 
*       term.
*************************************************************/
void
orth_poly_expansion_approx(double (*A)(double,void *), void *args, 
                           struct OrthPolyExpansion * poly)
{

    size_t ii, jj;
    double p[2];
    double pnew;

    /* double m = 1.0; */
    /* double off = 0.0; */

    double * fvals = NULL;
    double * pt_un = NULL; // unormalized point
    double * pt = NULL;
    double * wt = NULL; 

    size_t nquad = poly->num_poly+1;

    switch (poly->p->ptype) {
        case FOURIER:
            fprintf(stderr, "Cannot perform orth_poly_expansion_approx with Fourier basis\n");
            exit(1);
        case CHEBYSHEV:
            /* m = (poly->upper_bound - poly->lower_bound) /  */
            /*     (poly->p->upper - poly->p->lower); */
            /* off = poly->upper_bound - m * poly->p->upper; */
            pt = calloc_double(nquad);
            wt = calloc_double(nquad);
            cheb_gauss(poly->num_poly,pt,wt);

            /* clenshaw_curtis(nquad,pt,wt); */
            /* for (ii = 0; ii < nquad; ii++){wt[ii] *= 0.5;} */
            
            break;
        case LEGENDRE:
            /* m = (poly->upper_bound - poly->lower_bound) /  */
            /*     (poly->p->upper - poly->p->lower); */
            /* off = poly->upper_bound - m * poly->p->upper; */
//            nquad = poly->num_poly*2.0-1.0;//*2.0;
            pt = calloc_double(nquad);
            wt = calloc_double(nquad);
            
            // uncomment next two for cc
            // clenshaw_curtis(nquad,pt,wt);
//            for (ii = 0; ii < nquad; ii++){wt[ii] *= 0.5;}

            gauss_legendre(nquad,pt,wt);
            break;
        case HERMITE:
            pt = calloc_double(nquad);
            wt = calloc_double(nquad);
            gauss_hermite(nquad,pt,wt);
//            printf("point = ");
//            dprint(nquad,pt);
            break;
        case STANDARD:
            fprintf(stderr, "Cannot call orth_poly_expansion_approx for STANDARD type\n");
            break;
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", 
        //            poly->p->ptype);
    }
    
    fvals = calloc_double(nquad);
    pt_un = calloc_double(nquad);
    for (ii = 0; ii < nquad; ii++){
        /* pt_un[ii] =  m * pt[ii] + off; */
        pt_un[ii] = space_mapping_map_inverse(poly->space_transform,pt[ii]);
        fvals[ii] = A(pt_un[ii],args)  * wt[ii];
    }
    
    if (poly->num_poly > 1){
        for (ii = 0; ii < nquad; ii++){ // loop over all points
            p[0] = poly->p->const_term;
            poly->coeff[0] += fvals[ii] * poly->p->const_term;
            
            p[1] = poly->p->lin_const + poly->p->lin_coeff * pt[ii];
            poly->coeff[1] += fvals[ii] * p[1] ;
            // loop over all coefficients
            for (jj = 2; jj < poly->num_poly; jj++){ 
                pnew = eval_orth_poly_wp(poly->p, p[0], p[1], jj, pt[ii]);
                poly->coeff[jj] += fvals[ii] * pnew;
                p[0] = p[1];
                p[1] = pnew;
            }
        }

        for (ii = 0; ii < poly->num_poly; ii++){
            poly->coeff[ii] /= poly->p->norm(ii);
        }

    }
    else{
        for (ii = 0; ii < nquad; ii++){

            poly->coeff[0] += fvals[ii] *poly->p->const_term;
        }
        poly->coeff[0] /= poly->p->norm(0);
    }
    free(fvals); fvals = NULL;
    free(wt);    wt    = NULL;
    free(pt);    pt    = NULL;
    free(pt_un); pt_un = NULL;
    
}

/********************************************************//**
*  Construct an orthonormal polynomial expansion from (weighted) function 
*  evaluations and quadrature nodes
*  
*  \param[in,out] poly      - orthogonal polynomial expansion
*  \param[in]     num_nodes - number of nodes 
*  \param[in]     fvals     - function values (multiplied by a weight if necessary)
*  \param[in]     nodes     - locations of evaluations
*************************************************************/
static void
orth_poly_expansion_construct(struct OrthPolyExpansion * poly,
                              size_t num_nodes, double * fvals,
                              double * nodes)
{
    assert (poly->p->ptype != FOURIER);

    double p[2];
    double pnew;
    size_t ii,jj;
    if (poly->num_poly > 1){
        for (ii = 0; ii < num_nodes; ii++){ // loop over all points
            p[0] = poly->p->const_term;
            poly->coeff[0] += fvals[ii] * poly->p->const_term;
            p[1] = poly->p->lin_const + poly->p->lin_coeff * nodes[ii];
            poly->coeff[1] += fvals[ii] * p[1] ;
            // loop over all coefficients
            for (jj = 2; jj < poly->num_poly; jj++){
                pnew = eval_orth_poly_wp(poly->p, p[0], p[1], jj, nodes[ii]);
                poly->coeff[jj] += fvals[ii] * pnew;
                p[0] = p[1];
                p[1] = pnew;
            }
        }

        for (ii = 0; ii < poly->num_poly; ii++){
            poly->coeff[ii] /= poly->p->norm(ii);
        }

    }
    else{
        for (ii = 0; ii < num_nodes; ii++){
            poly->coeff[0] += fvals[ii] * poly->p->const_term;
        }
        poly->coeff[0] /= poly->p->norm(0);
    }
}

/********************************************************//**
*  Approximating a function that can take a vector of points as
*  input
*  
*  \param[in,out] poly - orthogonal polynomial expansion
*  \param[in]     f    - wrapped function
*  \param[in]     opts - approximation options
*
*  \return 0 - no problems, > 0 problem
*
*  \note  Maximum quadrature limited to 200 nodes
*************************************************************/
int
orth_poly_expansion_approx_vec(struct OrthPolyExpansion * poly,
                               struct Fwrap * f,
                               const struct OpeOpts * opts)    
{    
    assert (poly != NULL);
    assert (f != NULL);

    enum quad_rule qrule = C3_GAUSS_QUAD;
    size_t nquad = poly->num_poly+1;
    if (opts != NULL){
        qrule = opts->qrule;
        if (qrule == C3_CC_QUAD){
            nquad = 2*poly->num_poly-1;
        }
    }

    if ((nquad < 1 || nquad > 200) && (poly->p->ptype != FOURIER)){
        return 1;
    }

    double fvals[200];
    double pt_un[200];
    double qpt[200];
    double wt[200];
    
    double * quadpt = NULL;
    double * quadwt = NULL;

    int return_val = 0;
    switch (poly->p->ptype) {
    case FOURIER:
        return fourier_expansion_approx_vec(poly, f, opts);
    case CHEBYSHEV:
        assert (qrule == C3_GAUSS_QUAD);        
        return_val = cheb_gauss(nquad,qpt,wt);
        break;
    case LEGENDRE:
        if (qrule == C3_GAUSS_QUAD){
            return_val = getLegPtsWts2(nquad,&quadpt,&quadwt);
        }
        else if (qrule == C3_CC_QUAD){
            clenshaw_curtis(nquad,qpt,wt);
            for (size_t ii = 0; ii < nquad; ii++){wt[ii] *= 0.5;}
        }
        else{
            fprintf(stderr,"Specified quadrature rule not valid for legendre poly\n");
            exit(1);
        }
        break; 
    case HERMITE:
        assert (qrule == C3_GAUSS_QUAD);        
        return_val = gauss_hermite(nquad,qpt,wt);
        break;
    case STANDARD:
        fprintf(stderr, "Cannot call orth_poly_expansion_approx_vec");
        fprintf(stderr," for STANDARD type\n");
        return 1;
    }

    if (return_val != 0){
        return return_val;
    }

    // something other than legendre
    if (quadpt == NULL){ 
        quadpt = qpt;
        quadwt = wt;
    }
    
    for (size_t ii = 0; ii < nquad; ii++){ 
        pt_un[ii] = space_mapping_map_inverse(poly->space_transform,quadpt[ii]);
    }


    // Evaluate functions
    return_val = fwrap_eval(nquad,pt_un,fvals,f);
    if (return_val != 0){
        return return_val;
    }
    for (size_t ii = 0; ii < nquad; ii++){
        fvals[ii] *= quadwt[ii];
    }
    
    orth_poly_expansion_construct(poly,nquad,fvals,quadpt);

    /* printf("constructed\n"); */
    /* printf("pts = "); dprint(nquad, quadpt); */
    /* printf("vals = "); dprint(nquad, fvals); */
    /* for (size_t ii = 0; ii < nquad; ii++){ */
    /*     double peval = orth_poly_expansion_eval(poly, pt_un[ii]); */

    /*     printf("peval = %3.5G, fval = %3.5G\n", peval*quadwt[ii], fvals[ii]); */
    /* } */
    
    /* exit(1); */
    return return_val;
}

/********************************************************//**
*   Create an approximation adaptively
*
*   \param[in] aopts - approximation options
*   \param[in] fw    - wrapped function
*   
*   \return poly
*
*   \note 
*       Follows general scheme that trefethan outlines about 
*       Chebfun in his book Approximation Theory and practice
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_approx_adapt(const struct OpeOpts * aopts,
                                 struct Fwrap * fw)
{
    assert (aopts != NULL);
    assert (fw != NULL);
    if (aopts->ptype == FOURIER){

        size_t N = ope_opts_get_start(aopts);
        struct OrthPolyExpansion * poly = orth_poly_expansion_init_from_opts(aopts, N);
        orth_poly_expansion_approx_vec(poly, fw, NULL);

        return poly;
        /* fprintf(stderr, "Cannot perform approx_adapt with fourier basis\n"); */
        /* exit(1); */
    }
    
    size_t ii;
    size_t N = aopts->start_num;
    struct OrthPolyExpansion * poly = NULL;
    poly = orth_poly_expansion_init_from_opts(aopts,N);
    orth_poly_expansion_approx_vec(poly,fw,aopts);

    size_t coeffs_too_big = 0;
    for (ii = 0; ii < aopts->coeffs_check; ii++){
        if (fabs(poly->coeff[N-1-ii]) > aopts->tol){
            coeffs_too_big = 1;
            break;
        }
    }
    


    size_t maxnum = ope_opts_get_maxnum(aopts);
    /* printf("TOL SPECIFIED IS %G\n",aopts->tol); */
    /* printf("Ncoeffs check=%zu \n",aopts->coeffs_check); */
    /* printf("maxnum = %zu\n", maxnum); */
    while ((coeffs_too_big == 1) && (N < maxnum)){
        /* printf("N = %zu\n",N); */
        coeffs_too_big = 0;
	
        free(poly->coeff); poly->coeff = NULL;
        if (aopts->qrule == C3_CC_QUAD){
            N = N * 2 - 1; // for nested cc
        }
        else{
            N = N + 7;
        }
        poly->num_poly = N;
        poly->nalloc = N + OPECALLOC;
        poly->coeff = calloc_double(poly->nalloc);
//        printf("Number of coefficients to check = %zu\n",aopts->coeffs_check);
        orth_poly_expansion_approx_vec(poly,fw,aopts);
	    double sum_coeff_squared = 0.0;
        for (ii = 0; ii < N; ii++){ 
            sum_coeff_squared += pow(poly->coeff[ii],2); 
        }
        sum_coeff_squared = fmax(sum_coeff_squared,ZEROTHRESH);
        /* sum_coeff_squared = 1.0; */
        for (ii = 0; ii < aopts->coeffs_check; ii++){
            /* printf("aopts->tol=%3.15G last coefficients %3.15G\n", */
            /*        aopts->tol * sum_coeff_squared, */
           	/* 	  poly->coeff[N-1-ii]); */
            if (fabs(poly->coeff[N-1-ii]) > (aopts->tol * sum_coeff_squared) ){
                coeffs_too_big = 1;
                break;
            }
        }
        if (N > 100){
            //printf("Warning: num of poly is %zu: last coeff = %G \n",N,poly->coeff[N-1]);
            //printf("tolerance is %3.15G\n", aopts->tol * sum_coeff_squared);
            //printf("Considering using piecewise polynomials\n");
            /*
            printf("first 5 coeffs\n");

            size_t ll;
            for (ll = 0; ll<5;ll++){
                printf("%3.10G ",poly->coeff[ll]);
            }
            printf("\n");

            printf("Last 10 coeffs\n");
            for (ll = 0; ll<10;ll++){
                printf("%3.10G ",poly->coeff[N-10+ll]);
            }
            printf("\n");
            */
            coeffs_too_big = 0;
        }

    }
    
    orth_poly_expansion_round(&poly);

    // verify
    /* double pt = (upper - lower)*randu() + lower; */
    /* double val_true = A(pt,args); */
    /* double val_test = orth_poly_expansion_eval(poly,pt); */
    /* double diff = val_true-val_test; */
    /* double err = fabs(diff); */
    /* if (fabs(val_true) > 1.0){ */
    /* //if (fabs(val_true) > ZEROTHRESH){ */
    /*     err /= fabs(val_true); */
    /* } */
    /* if (err > 100.0*aopts->tol){ */
    /*     //fprintf(stderr, "Approximating at point %G in (%3.15G,%3.15G)\n",pt,lower,upper); */
    /*     //fprintf(stderr, "leads to error %G, while tol is %G \n",err,aopts->tol); */
    /*     //fprintf(stderr, "actual value is %G \n",val_true); */
    /*     //fprintf(stderr, "predicted value is %3.15G \n",val_test); */
    /*     //fprintf(stderr, "%zu N coeffs, last coeffs are %3.15G,%3.15G \n",N,poly->coeff[N-2],poly->coeff[N-1]); */
    /*     //exit(1); */
    /* } */

    /* if (default_opts == 1){ */
    /*     ope_opts_free(aopts); */
    /* } */
    return poly;
}

/********************************************************//**
*   Generate an orthonormal polynomial with pseudorandom coefficients
*   between [-1,1]
*
*   \param[in] ptype    - polynomial type
*   \param[in] maxorder - maximum order of the polynomial
*   \param[in] lower    - lower bound of input
*   \param[in] upper    - upper bound of input
*
*   \return poly
*************************************************************/
struct OrthPolyExpansion * 
orth_poly_expansion_randu(enum poly_type ptype, size_t maxorder, 
                          double lower, double upper)
{
    struct OrthPolyExpansion * poly =
        orth_poly_expansion_init(ptype,maxorder+1, lower, upper);
                                        
    size_t ii;
    for (ii = 0; ii < poly->num_poly; ii++){
        poly->coeff[ii] = randu()*2.0-1.0;
    }
    return poly;
}

/********************************************************//**
*   Integrate a Chebyshev approximation
*
*   \param[in] poly - polynomial to integrate
*
*   \return out - integral of approximation
*************************************************************/
double
cheb_integrate2(const struct OrthPolyExpansion * poly)
{
    size_t ii;
    double out = 0.0;
    
    double m = space_mapping_map_inverse_deriv(poly->space_transform,0.0);
    /* double m =  */
    for (ii = 0; ii < poly->num_poly; ii+=2){
        out += poly->coeff[ii] * 2.0 / (1.0 - (double) (ii*ii));
    }
    out = out * m;
    return out;
}

/********************************************************//**
*   Integrate a Legendre approximation
*
*   \param[in] poly - polynomial to integrate
*
*   \return out - integral of approximation
*************************************************************/
double
legendre_integrate(const struct OrthPolyExpansion * poly)
{
    double out = 0.0;

    double m = space_mapping_map_inverse_deriv(poly->space_transform,0.0);
    out = poly->coeff[0] * 2.0;
    out = out * m;
    return out;
}

/********************************************************//**
*   Compute the product of two polynomial expansion
*
*   \param[in] a - first polynomial
*   \param[in] b - second polynomial
*
*   \return c - polynomial expansion
*
*   \note 
*   Computes c(x) = a(x)b(x) where c is same form as a
*   Lower and upper bounds of both polynomials must be the same
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_prod(const struct OrthPolyExpansion * a,
                         const struct OrthPolyExpansion * b)
{
    
    struct OrthPolyExpansion * c = NULL;
    double lb = a->lower_bound;
    double ub = a->upper_bound;
    assert ( fabs(a->lower_bound - b->lower_bound) < DBL_EPSILON );
    assert ( fabs(a->upper_bound - b->upper_bound) < DBL_EPSILON );
    
    enum poly_type p = a->p->ptype;
    if (p == FOURIER){
        return fourier_expansion_prod(a, b);
    }
    if ( (p == LEGENDRE) && (a->num_poly < 100) && (b->num_poly < 100)){
        //printf("in special prod\n");
        //double lb = a->lower_bound;
        //double ub = b->upper_bound;
            
        size_t ii,jj;
        c = orth_poly_expansion_init(p, a->num_poly + b->num_poly+1, lb, ub);
        double * allprods = calloc_double(a->num_poly * b->num_poly);
        for (ii = 0; ii < a->num_poly; ii++){
            for (jj = 0; jj < b->num_poly; jj++){
                allprods[jj + ii * b->num_poly] = a->coeff[ii] * b->coeff[jj];
            }
        }
        
        //printf("A = \n");
        //print_orth_poly_expansion(a,1,NULL);

        //printf("B = \n");
        //print_orth_poly_expansion(b,1,NULL);

        //dprint2d_col(b->num_poly, a->num_poly, allprods);

        size_t kk;
        for (kk = 0; kk < c->num_poly; kk++){
            for (ii = 0; ii < a->num_poly; ii++){
                for (jj = 0; jj < b->num_poly; jj++){
                    c->coeff[kk] +=  lpolycoeffs[ii+jj*100+kk*10000] * 
                                        allprods[jj+ii*b->num_poly];
                }
            }
            //printf("c coeff[%zu]=%G\n",kk,c->coeff[kk]);
        }
        orth_poly_expansion_round(&c);
        free(allprods); allprods=NULL;
    }
    /* else if (p == CHEBYSHEV){ */
    /*     c = orth_poly_expansion_init(p,a->num_poly+b->num_poly+1,lb,ub); */
    /*     for (size_t ii = 0; ii) */
    /* } */
    else{
//        printf("OrthPolyExpansion product greater than order 100 is slow\n");
        const struct OrthPolyExpansion * comb[2];
        comb[0] = a;
        comb[1] = b;
        
        double norma = 0.0, normb = 0.0;
        size_t ii;
        for (ii = 0; ii < a->num_poly; ii++){
            norma += pow(a->coeff[ii],2);
        }
        for (ii = 0; ii < b->num_poly; ii++){
            normb += pow(b->coeff[ii],2);
        }
        
        if ( (norma < ZEROTHRESH) || (normb < ZEROTHRESH) ){
            //printf("in here \n");
//            c = orth_poly_expansion_constant(0.0,a->p->ptype,lb,ub);
            c = orth_poly_expansion_init(p,1, lb, ub);
            space_mapping_free(c->space_transform); c->space_transform = NULL;
            c->space_transform = space_mapping_copy(a->space_transform);
            c->coeff[0] = 0.0;
        }
        else{
            //printf(" total order of product = %zu\n",a->num_poly+b->num_poly);
            c = orth_poly_expansion_init(p, a->num_poly + b->num_poly+5, lb, ub);
            space_mapping_free(c->space_transform); c->space_transform = NULL;
            c->space_transform = space_mapping_copy(a->space_transform);
            orth_poly_expansion_approx(&orth_poly_expansion_eval3,comb,c);
            /* printf("num_coeff pre_round = %zu\n", c->num_poly); */
            /* orth_poly_expansion_approx(&orth_poly_expansion_eval3,comb,c); */
            orth_poly_expansion_round(&c);
            /* printf("num_coeff post_round = %zu\n", c->num_poly); */
        }
    }
    
    //*
    //printf("compute product\n");
    //struct OpeOpts ao;
    //ao.start_num = 3;
    //ao.coeffs_check = 2;
    //ao.tol = 1e-13;
    //c = orth_poly_expansion_approx_adapt(&orth_poly_expansion_eval3,comb, 
    //                    p, lb, ub, &ao);
    
    //orth_poly_expansion_round(&c);
    //printf("done\n");
    //*/
    return c;
}

/********************************************************//**
*   Compute the sum of the product between the functions in two arraysarrays
*
*   \param[in] n   - number of functions
*   \param[in] lda - stride of first array
*   \param[in] a   - array of orthonormal polynomial expansions
*   \param[in] ldb - stride of second array
*   \param[in] b   - array of orthonormal polynomial expansions
*
*   \return c - polynomial expansion
*
*   \note 
*       All the functions need to have the same lower 
*       and upper bounds and be of the same type
*
*       If the maximum order of the polynomials is greater than 25 then this is
*       inefficient because I haven't precomputed triple product rules
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_sum_prod(size_t n, size_t lda, 
                             struct OrthPolyExpansion ** a, size_t ldb,
                             struct OrthPolyExpansion ** b)
{

    enum poly_type ptype;
    ptype = a[0]->p->ptype;
    struct OrthPolyExpansion * c = NULL;
    double lb = a[0]->lower_bound;
    double ub = a[0]->upper_bound;

    size_t ii;
    size_t maxordera = 0;
    size_t maxorderb = 0;
    size_t maxorder = 0;
    //int legen = 1;
    for (ii = 0; ii < n; ii++){

        if (a[ii*lda]->p->ptype !=  b[ii*ldb]->p->ptype){
            return c; // cant do it
        }
        else if (a[ii*lda]->p->ptype != ptype){
            return c;
        }
        size_t neworder = a[ii*lda]->num_poly + b[ii*ldb]->num_poly;
        if (neworder > maxorder){
            maxorder = neworder;
        }
        if (a[ii*lda]->num_poly > maxordera){
            maxordera = a[ii*lda]->num_poly;
        }
        if (b[ii*ldb]->num_poly > maxorderb){
            maxorderb = b[ii*ldb]->num_poly;
        }
    }
    if ( (maxordera > 99) || (maxorderb > 99) || (ptype != LEGENDRE)){
        printf("OrthPolyExpansion sum_product greater than order 100 is slow\n");
        c = orth_poly_expansion_prod(a[0],b[0]);
        for (ii = 1; ii< n; ii++){
            struct OrthPolyExpansion * temp = 
                orth_poly_expansion_prod(a[ii*lda],b[ii*ldb]);
            orth_poly_expansion_axpy(1.0,temp,c);
            orth_poly_expansion_free(temp); 
            temp = NULL;
        }
    }
    else{
        enum poly_type p = LEGENDRE;
        c = orth_poly_expansion_init(p, maxorder, lb, ub);
        size_t kk,jj,ll;
        double * allprods = calloc_double( maxorderb * maxordera);
        for (kk = 0; kk < n; kk++){
            for (ii = 0; ii < a[kk*lda]->num_poly; ii++){
                for (jj = 0; jj < b[kk*ldb]->num_poly; jj++){
                    allprods[jj + ii * maxorderb] += 
                            a[kk*lda]->coeff[ii] * b[kk*ldb]->coeff[jj];
                }
            }
        }

        for (ll = 0; ll < c->num_poly; ll++){
            for (ii = 0; ii < maxordera; ii++){
                for (jj = 0; jj < maxorderb; jj++){
                    c->coeff[ll] +=  lpolycoeffs[ii+jj*100+ll*10000] * 
                                        allprods[jj+ii*maxorderb];
                }
            }
        }
        free(allprods); allprods=NULL;
        orth_poly_expansion_round(&c);
    }
    return c;
}

/********************************************************//**
*   Compute a linear combination of generic functions
*
*   \param[in] n   - number of functions
*   \param[in] ldx - stride of first array
*   \param[in] x   - functions
*   \param[in] ldc - stride of coefficients
*   \param[in] c   - scaling coefficients
*
*   \return  out  = \f$ \sum_{i=1}^n coeff[ldc[i]] * gfa[ldgf[i]] \f$
*
*   \note 
*       If both arrays do not consist of only LEGENDRE polynomials
*       return NULL. All the functions need to have the same lower 
*       and upper bounds
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_lin_comb(size_t n, size_t ldx, 
                             struct OrthPolyExpansion ** x, size_t ldc,
                             const double * c )
{

    struct OrthPolyExpansion * out = NULL;
    double lb = x[0]->lower_bound;
    double ub = x[0]->upper_bound;
    enum poly_type ptype = x[0]->p->ptype;
    size_t ii;
    size_t maxorder = 0;
    //int legen = 1;
    for (ii = 0; ii < n; ii++){
        if (x[ii*ldx]->p->ptype != ptype){
            //legen = 0;
            return out; // cant do it
        }
        size_t neworder = x[ii*ldx]->num_poly;
        if (neworder > maxorder){
            maxorder = neworder;
        }
    }
    

    out = orth_poly_expansion_init(ptype, maxorder, lb, ub);
    space_mapping_free(out->space_transform);
    out->space_transform = space_mapping_copy(x[0]->space_transform);
    size_t kk;
    if (ptype != FOURIER){
        for (kk = 0; kk < n; kk++){
            for (ii = 0; ii < x[kk*ldx]->num_poly; ii++){
                out->coeff[ii] +=  c[kk*ldc]*x[kk*ldx]->coeff[ii];
            }
        }
    }
    else{
        for (kk = 0; kk < n; kk++){
            for (ii = 0; ii < x[kk*ldx]->num_poly; ii++){
                out->ccoeff[ii] +=  c[kk*ldc]*x[kk*ldx]->ccoeff[ii];
            }
        }
    }
    orth_poly_expansion_round(&out);
    return out;
}

/********************************************************//**
*   Integrate an orthogonal polynomial expansion 
*
*   \param[in] poly - polynomial to integrate
*
*   \return out - Integral of approximation
*
*   \note 
*       Need to an 'else' or default behavior to switch case
*       int_{lb}^ub  f(x) dx
*    For Hermite polynomials this integrates with respec to
*    the Gaussian weight
*************************************************************/
double
orth_poly_expansion_integrate(const struct OrthPolyExpansion * poly)
{
    double out = 0.0;
    switch (poly->p->ptype){
    case LEGENDRE:  out = legendre_integrate(poly); break;
    case HERMITE:   out = hermite_integrate(poly);  break;
    case CHEBYSHEV: out = cheb_integrate2(poly);    break;
    case FOURIER:   out = fourier_integrate(poly);  break;        
    case STANDARD:  fprintf(stderr, "Cannot integrate STANDARD type\n"); break;
    }
    return out;
}

/********************************************************//**
*   Integrate an orthogonal polynomial expansion 
*
*   \param[in] poly - polynomial to integrate
*
*   \return out - Integral of approximation
*
    \note Computes  \f$ \int f(x) w(x) dx \f$ for every univariate function
    in the qmarray
    
    w(x) depends on underlying parameterization
    for example, it is 1/2 for legendre (and default for others),
    gauss for hermite,etc
*************************************************************/
double
orth_poly_expansion_integrate_weighted(const struct OrthPolyExpansion * poly)
{
    double out = 0.0;
    switch (poly->p->ptype){
    case LEGENDRE:  out = poly->coeff[0];  break;
    case HERMITE:   out = poly->coeff[0];  break;
    case CHEBYSHEV: out = poly->coeff[0];  break;
    case FOURIER:   out = creal(poly->ccoeff[0]); break;
    case STANDARD:  fprintf(stderr, "Cannot integrate STANDARD type\n"); break;
    }

    return out;
}


/********************************************************//**
*   Weighted inner product between two polynomial 
*   expansions of the same type
*
*   \param[in] a - first polynomial
*   \param[in] b - second polynomai
*
*   \return inner product
*
*   \note
*       Computes \f[ \int_{lb}^ub  a(x)b(x) w(x) dx \f]
*
*************************************************************/
double
orth_poly_expansion_inner_w(const struct OrthPolyExpansion * a,
                            const struct OrthPolyExpansion * b)
{
    assert(a->p->ptype == b->p->ptype);
    assert (a->p->ptype != FOURIER);
    double out = 0.0;
    size_t N = a->num_poly < b->num_poly ? a->num_poly : b->num_poly;
    size_t ii;
    for (ii = 0; ii < N; ii++){
        out += a->coeff[ii] * b->coeff[ii] * a->p->norm(ii); 
    }

    return out;
}

/********************************************************//**
*   Inner product between two polynomial expansions of the same type
*
*   \param[in] a - first polynomial
*   \param[in] b - second polynomai
*
*   \return  inner product
*
*   \note
*   If the polynomial is NOT HERMITE then
*   Computes  \f$ \int_{lb}^ub  a(x)b(x) dx \f$ by first
*   converting each polynomial to a Legendre polynomial
*   Otherwise it computes the error with respect to gaussia weight
*************************************************************/
double
orth_poly_expansion_inner(const struct OrthPolyExpansion * a,
                          const struct OrthPolyExpansion * b)
{   
    struct OrthPolyExpansion * t1 = NULL;
    struct OrthPolyExpansion * t2 = NULL;
    
//    assert (a->ptype == b->ptype);
//    enum poly_type ptype = a->ptype;
//    switch (ptype)
    if ((a->p->ptype == HERMITE) && (b->p->ptype == HERMITE)){
        return orth_poly_expansion_inner_w(a,b);
    }
    else if ((a->p->ptype == CHEBYSHEV) && (b->p->ptype == CHEBYSHEV)){

        struct OrthPolyExpansion * prod = orth_poly_expansion_prod(a, b);
        double int_val = orth_poly_expansion_integrate(prod);
        orth_poly_expansion_free(prod); prod = NULL;
        return int_val;
        /* // can possibly make this more efficient */
        /* double out = 0.0; */
        /* size_t N = a->num_poly < b->num_poly ? a->num_poly : b->num_poly; */
        /* for (size_t ii = 0; ii < N; ii++){ */
        /*     for (size_t jj = 0; jj < ii; jj++){ */
        /*         if ( ((ii+jj) % 2) == 0){ */
        /*             out += (a->coeff[ii]*b->coeff[jj] *  */
        /*                      (1.0 / (1.0 - (double) (ii-jj)*(ii-jj)) */
        /*                       + 1.0 / (1.0 - (double) (ii+jj)*(ii+jj)))); */
        /*         } */
        /*     } */
        /*     for (size_t jj = ii; jj < N; jj++){ */
        /*         if ( ((ii+jj) % 2) == 0){ */
        /*             out += (a->coeff[ii]*b->coeff[jj] *  */
        /*                      (1.0 / (1.0 - (double) (jj-ii)*(jj-ii)) */
        /*                       + 1.0 / (1.0 - (double) (ii+jj)*(ii+jj)))); */
        /*         } */
        /*     } */
        /* } */
        /* double m = (a->upper_bound - a->lower_bound) / (a->p->upper - a->p->lower); */
        /* out *=  m; */
        /* return out */;
    }
    else if ((a->p->ptype == FOURIER) && (b->p->ptype == FOURIER)){
        struct OrthPolyExpansion * prod = orth_poly_expansion_prod(a, b);
        double int_val = orth_poly_expansion_integrate(prod);
        orth_poly_expansion_free(prod); prod = NULL;
        return int_val;
    }
    else{
        if (a->p->ptype == CHEBYSHEV){
            t1 = orth_poly_expansion_init(LEGENDRE, a->num_poly,
                                          a->lower_bound, a->upper_bound);
            orth_poly_expansion_approx(&orth_poly_expansion_eval2, (void*)a, t1);
            orth_poly_expansion_round(&t1);
        }
        else if (a->p->ptype != LEGENDRE){
            fprintf(stderr, "Don't know how to take inner product using polynomial type. \n");
            fprintf(stderr, "type1 = %d, and type2= %d\n",a->p->ptype,b->p->ptype);
            exit(1);
        }

        if (b->p->ptype == CHEBYSHEV){
            t2 = orth_poly_expansion_init(LEGENDRE, b->num_poly,
                                          b->lower_bound, b->upper_bound);
            orth_poly_expansion_approx(&orth_poly_expansion_eval2, (void*)b, t2);
            orth_poly_expansion_round(&t2);
        }
        else if (b->p->ptype != LEGENDRE){
            fprintf(stderr, "Don't know how to take inner product using polynomial type. \n");
            fprintf(stderr, "type1 = %d, and type2= %d\n",a->p->ptype,b->p->ptype);
            exit(1);
        }

        double out;
        if ((t1 == NULL) && (t2 == NULL)){
            out = orth_poly_expansion_inner_w(a,b) * (a->upper_bound - a->lower_bound);
        }
        else if ((t1 == NULL) && (t2 != NULL)){
            out = orth_poly_expansion_inner_w(a,t2) * (a->upper_bound - a->lower_bound);
            orth_poly_expansion_free(t2); t2 = NULL;
        }
        else if ((t2 == NULL) && (t1 != NULL)){
            out = orth_poly_expansion_inner_w(t1,b) * (b->upper_bound - b->lower_bound);
            orth_poly_expansion_free(t1); t1 = NULL;
        }
        else{
            out = orth_poly_expansion_inner_w(t1,t2) * (t1->upper_bound - t1->lower_bound);
            orth_poly_expansion_free(t1); t1 = NULL;
            orth_poly_expansion_free(t2); t2 = NULL;
        }
        return out;
    }
}

/********************************************************//**
*   Compute the norm of an orthogonal polynomial
*   expansion with respect to family weighting 
*   function
*
*   \param[in] p - polynomial to integrate
*
*   \return out - norm of function
*
*   \note
*        Computes  \f$ \sqrt(\int f(x)^2 w(x) dx) \f$
*************************************************************/
double orth_poly_expansion_norm_w(const struct OrthPolyExpansion * p){

    double out = sqrt(orth_poly_expansion_inner_w(p,p));
    return sqrt(out);
}

/********************************************************//**
*   Compute the norm of an orthogonal polynomial
*   expansion with respect to uniform weighting 
*   (except in case of HERMITE, then do gaussian weighting)
*
*   \param[in] p - polynomial of which to obtain norm
*
*   \return out - norm of function
*
*   \note
*        Computes \f$ \sqrt(\int_a^b f(x)^2 dx) \f$
*************************************************************/
double orth_poly_expansion_norm(const struct OrthPolyExpansion * p){

    double out = 0.0;
    out = sqrt(orth_poly_expansion_inner(p,p));
    return out;
}

/********************************************************//**
*   Multiply polynomial expansion by -1
*
*   \param[in,out] p - polynomial multiply by -1
*************************************************************/
void 
orth_poly_expansion_flip_sign(struct OrthPolyExpansion * p)
{   

    if (p->p->ptype != FOURIER){
        for (size_t ii = 0; ii < p->num_poly; ii++){
            p->coeff[ii] *= -1.0;
        }
    }
    else{
        for (size_t ii = 0; ii < p->num_poly; ii++){
            p->ccoeff[ii] *= -1.0;
        }
    }
}

/********************************************************//**
*   Multiply by scalar and overwrite expansion
*
*   \param[in] a - scaling factor for first polynomial
*   \param[in] x - polynomial to scale
*************************************************************/
void orth_poly_expansion_scale(double a, struct OrthPolyExpansion * x)
{
    
    size_t ii;
    if (x->p->ptype != FOURIER){
        for (ii = 0; ii < x->num_poly; ii++){
            x->coeff[ii] *= a;
        }
    }
    else{
        for (ii = 0; ii < x->num_poly; ii++){
            x->ccoeff[ii] *= a;
        }
    }
    orth_poly_expansion_round(&x);
}

/********************************************************//**
*   Multiply and add 3 expansions \f$ z \leftarrow ax + by + cz \f$
*
*   \param[in] a  - scaling factor for first polynomial
*   \param[in] x  - first polynomial
*   \param[in] b  - scaling factor for second polynomial
*   \param[in] y  - second polynomial
*   \param[in] c  - scaling factor for third polynomial
*   \param[in] z  - third polynomial
*
*************************************************************/
void
orth_poly_expansion_sum3_up(double a, struct OrthPolyExpansion * x,
                           double b, struct OrthPolyExpansion * y,
                           double c, struct OrthPolyExpansion * z)
{
    assert (x->p->ptype == y->p->ptype);
    assert (y->p->ptype == z->p->ptype);
    assert ( x != NULL );
    assert ( y != NULL );
    assert ( z != NULL );
    
    assert (x->p->ptype != FOURIER);
    
    size_t ii;
    if ( (z->num_poly >= x->num_poly) && (z->num_poly >= y->num_poly) ){
        
        if (x->num_poly > y->num_poly){
            for (ii = 0; ii < y->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii] + b*y->coeff[ii];
            }
            for (ii = y->num_poly; ii < x->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii];
            }
            for (ii = x->num_poly; ii < z->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii];
            }
        }
        else{
            for (ii = 0; ii < x->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii] + b*y->coeff[ii];
            }
            for (ii = x->num_poly; ii < y->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii] + b*y->coeff[ii];
            }
            for (ii = x->num_poly; ii < z->num_poly; ii++){
                z->coeff[ii] = c*z->coeff[ii];
            }
        }
    }
    else if ((z->num_poly >= x->num_poly) && ( z->num_poly < y->num_poly)) {
        double * temp = realloc(z->coeff, (y->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        else{
            z->coeff = temp;
        }
        for (ii = 0; ii < x->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = x->num_poly; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii] + b*y->coeff[ii];
        }
        for (ii = z->num_poly; ii < y->num_poly; ii++){
            z->coeff[ii] = b*y->coeff[ii];
        }
        z->num_poly = y->num_poly;
    }
    else if ( (z->num_poly < x->num_poly) && ( z->num_poly >= y->num_poly) ){
        double * temp = realloc(z->coeff, (x->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        else{
            z->coeff = temp;
        }
        for (ii = 0; ii < y->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = y->num_poly; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii] + a*x->coeff[ii];
        }
        for (ii = z->num_poly; ii < x->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii];
        }
        z->num_poly = x->num_poly;
    }
    else if ( x->num_poly <= y->num_poly){
        double * temp = realloc(z->coeff, (y->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        for (ii = 0; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = z->num_poly; ii < x->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii] + b*y->coeff[ii];
        }
        for (ii = x->num_poly; ii < y->num_poly; ii++){
            z->coeff[ii] = b*y->coeff[ii];
        }
        z->num_poly = y->num_poly;
    }
    else if (y->num_poly <= x->num_poly) {
        double * temp = realloc(z->coeff, (x->num_poly)*sizeof(double));
        if (temp == NULL){
            fprintf(stderr,"cannot allocate new size fo z-coeff in sum3_up\n");
            exit(1);
        }
        for (ii = 0; ii < z->num_poly; ii++){
            z->coeff[ii] = c*z->coeff[ii]+a*x->coeff[ii]+b*y->coeff[ii];
        }
        for (ii = z->num_poly; ii < y->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii] + b*y->coeff[ii];
        }
        for (ii = y->num_poly; ii < x->num_poly; ii++){
            z->coeff[ii] = a*x->coeff[ii];
        }
        z->num_poly = x->num_poly;
    }
    else{
        fprintf(stderr,"Haven't accounted for anything else?! %zu %zu %zu\n", 
                x->num_poly, y->num_poly, z->num_poly);
        exit(1);
    }
    //   z->nalloc = z->num_poly + OPECALLOC;
    orth_poly_expansion_round(&z);
}

/********************************************************//**
*   Multiply by scalar and add two orthgonal 
*   expansions of the same family together \f[ y \leftarrow ax + y \f]
*
*   \param[in] a  - scaling factor for first polynomial
*   \param[in] x  - first polynomial
*   \param[in] y  - second polynomial
*
*   \return 0 if successfull 1 if error with allocating more space for y
*
*   \note 
*       Computes z=ax+by, where x and y are polynomial expansionx
*       Requires both polynomials to have the same upper 
*       and lower bounds
*       
**************************************************************/
int orth_poly_expansion_axpy(double a, struct OrthPolyExpansion * x,
                             struct OrthPolyExpansion * y)
{
    assert (y != NULL);
    assert (x != NULL);
    assert (x->p->ptype == y->p->ptype);
    
    assert ( fabs(x->lower_bound - y->lower_bound) < DBL_EPSILON );
    assert ( fabs(x->upper_bound - y->upper_bound) < DBL_EPSILON );

    if (x->p->ptype == FOURIER){
        return fourier_expansion_axpy(a, x, y);
    }
        
    if (x->num_poly < y->num_poly){
        // shouldnt need rounding here
        size_t ii;
        for (ii = 0; ii < x->num_poly; ii++){
            y->coeff[ii] += a * x->coeff[ii];
            if (fabs(y->coeff[ii]) < ZEROTHRESH){
                y->coeff[ii] = 0.0;
            }
        }
    }
    else{
        size_t ii;
        if (x->num_poly > y->nalloc){
            //printf("hereee\n");
            y->nalloc = x->num_poly+10;
            double * temp = realloc(y->coeff, (y->nalloc)*sizeof(double));
            if (temp == NULL){
                return 0;
            }
            else{
                y->coeff = temp;
                for (ii = y->num_poly; ii < y->nalloc; ii++){
                    y->coeff[ii] = 0.0;
                }
            }
            //printf("finished\n");
        }
        for (ii = y->num_poly; ii < x->num_poly; ii++){
            y->coeff[ii] = a * x->coeff[ii];
            if (fabs(y->coeff[ii]) < ZEROTHRESH){
                y->coeff[ii] = 0.0;
            }
        }
        for (ii = 0; ii < y->num_poly; ii++){
            y->coeff[ii] += a * x->coeff[ii];
            if (fabs(y->coeff[ii]) < ZEROTHRESH){
                y->coeff[ii] = 0.0;
            }
        }
        y->num_poly = x->num_poly;
        size_t nround = y->num_poly;
        for (ii = 0; ii < y->num_poly-1;ii++){
            if (fabs(y->coeff[y->num_poly-1-ii]) > ZEROTHRESH){
                break;
            }
            else{
                nround = nround-1;
            }
        }
        y->num_poly = nround;
    }
    
    return 0;
}


/********************************************************//**
*   Multiply by scalar and add two orthgonal 
*   expansions of the same family together
*
*   \param[in] a - scaling factor for first polynomial
*   \param[in] x - first polynomial
*   \param[in] b - scaling factor for second polynomial
*   \param[in] y  second polynomial
*
*   \return p - orthogonal polynomial expansion
*
*   \note 
*       Computes z=ax+by, where x and y are polynomial expansionx
*       Requires both polynomials to have the same upper 
*       and lower bounds
*   
*************************************************************/
struct OrthPolyExpansion *
orth_poly_expansion_daxpby(double a, struct OrthPolyExpansion * x,
                           double b, struct OrthPolyExpansion * y)
{
    /* assert (x->p->ptype != FOURIER); */
    struct OrthPolyExpansion * p ;
    if (x != NULL && y != NULL){
        p = orth_poly_expansion_copy(y);
        orth_poly_expansion_scale(b, p);
        orth_poly_expansion_axpy(a, x, p);
    }
    else if (x != NULL){
        p = orth_poly_expansion_copy(x);
        orth_poly_expansion_scale(a, p);
    }
    else if (y != NULL){
        p = orth_poly_expansion_copy(y);
        orth_poly_expansion_scale(b, p);
    }
    else{
        fprintf(stderr, "Error in orth_poly_expansion_daxpby\n");
        fprintf(stderr, "trying to run with all NULL arguments\n");
        exit(1);
    }
    return p;
}

////////////////////////////////////////////////////////////////////////////
// Algorithms

/********************************************************//**
*   Obtain the real roots of a standard polynomial
*
*   \param[in]     p     - standard polynomial
*   \param[in,out] nkeep - returns how many real roots tehre are
*
*   \return real_roots - real roots of a standard polynomial
*
*   \note
*   Only roots within the bounds are returned
*************************************************************/
double *
standard_poly_real_roots(struct StandardPoly * p, size_t * nkeep)
{
    if (p->num_poly == 1) // constant function
    {   
        double * real_roots = NULL;
        *nkeep = 0;
        return real_roots;
    }
    else if (p->num_poly == 2){ // linear
        double root = -p->coeff[0] / p->coeff[1];
        
        if ((root > p->lower_bound) && (root < p->upper_bound)){
            *nkeep = 1;
        }
        else{
            *nkeep = 0;
        }
        double * real_roots = NULL;
        if (*nkeep == 1){
            real_roots = calloc_double(1);
            real_roots[0] = root;
        }
        return real_roots;
    }
    
    size_t nrows = p->num_poly-1;
    //printf("coeffs = \n");
    //dprint(p->num_poly, p->coeff);
    while (fabs(p->coeff[nrows]) < ZEROTHRESH ){
    //while (fabs(p->coeff[nrows]) < DBL_MIN){
        nrows--;
        if (nrows == 1){
            break;
        }
    }

    //printf("nrows left = %zu \n",  nrows);
    if (nrows == 1) // linear
    {
        double root = -p->coeff[0] / p->coeff[1];
        if ((root > p->lower_bound) && (root < p->upper_bound)){
            *nkeep = 1;
        }
        else{
            *nkeep = 0;
        }
        double * real_roots = NULL;
        if (*nkeep == 1){
            real_roots = calloc_double(1);
            real_roots[0] = root;
        }
        return real_roots;
    }
    else if (nrows == 0)
    {
        double * real_roots = NULL;
        *nkeep = 0;
        return real_roots;
    }

    // transpose of the companion matrix
    double * t_companion = calloc_double((p->num_poly-1)*(p->num_poly-1));
    size_t ii;
    

   // size_t m = nrows;
    t_companion[nrows-1] = -p->coeff[0]/p->coeff[nrows];
    for (ii = 1; ii < nrows; ii++){
        t_companion[ii * nrows + ii-1] = 1.0;
        t_companion[ii * nrows + nrows-1] = -p->coeff[ii]/p->coeff[nrows];
    }
    double * real = calloc_double(nrows);
    double * img = calloc_double(nrows);
    int info;
    int lwork = 8 * nrows;
    double * iwork = calloc_double(8 * nrows);
    //double * vl;
    //double * vr;
    int n = nrows;

    //printf("hello! n=%d \n",n);
    dgeev_("N","N", &n, t_companion, &n, real, img, NULL, &n,
           NULL, &n, iwork, &lwork, &info);
    
    //printf("info = %d", info);

    free (iwork);
    
    int * keep = calloc_int(nrows);
    *nkeep = 0;
    // the 1e-10 is kinda hacky
    for (ii = 0; ii < nrows; ii++){
        //printf("real[ii] - p->lower_bound = %G\n",real[ii]-p->lower_bound);
        //printf("real root = %3.15G, imag = %G \n",real[ii],img[ii]);
        //printf("lower thresh = %3.20G\n",p->lower_bound-1e-8);
        //printf("zero thresh = %3.20G\n",1e-8);
        //printf("upper thresh = %G\n",p->upper_bound+ZEROTHRESH);
        //printf("too low? %d \n", real[ii] < (p->lower_bound-1e-8));
        if ((fabs(img[ii]) < 1e-7) && 
            (real[ii] > (p->lower_bound-1e-8)) && 
            //(real[ii] >= (p->lower_bound-1e-7)) && 
            (real[ii] < (p->upper_bound+1e-8))) {
            //(real[ii] <= (p->upper_bound+1e-7))) {
        
            //*
            if (real[ii] < p->lower_bound){
                real[ii] = p->lower_bound;
            }
            if (real[ii] > p->upper_bound){
                real[ii] = p->upper_bound;
            }
            //*/

            keep[ii] = 1;
            *nkeep = *nkeep + 1;
            //printf("keep\n");
        }
        else{
            keep[ii] = 0;
        }
    }
    
    /*
    printf("real portions roots = ");
    dprint(nrows, real);
    printf("imag portions roots = ");
    for (ii = 0; ii < nrows; ii++) printf("%E ",img[ii]);
    printf("\n");
    //dprint(nrows, img);
    */

    double * real_roots = calloc_double(*nkeep);
    size_t counter = 0;
    for (ii = 0; ii < nrows; ii++){
        if (keep[ii] == 1){
            real_roots[counter] = real[ii];
            counter++;
        }

    }
    
    free(t_companion);
    free(real);
    free(img);
    free(keep);

    return real_roots;
}

static int dblcompare(const void * a, const void * b)
{
    const double * aa = a;
    const double * bb = b;
    if ( *aa < *bb){
        return -1;
    }
    return 1;
}

/********************************************************//**
*   Obtain the real roots of a legendre polynomial expansion
*
*   \param[in]     p     - orthogonal polynomial expansion
*   \param[in,out] nkeep - returns how many real roots tehre are
*
*   \return real_roots - real roots of an orthonormal polynomial expansion
*
*   \note
*       Only roots within the bounds are returned
*       Algorithm is based on eigenvalues of non-standard companion matrix from
*       Roots of Polynomials Expressed in terms of orthogonal polynomials
*       David Day and Louis Romero 2005
*
*       Multiplying by a factor of sqrt(2*N+1) because using orthonormal,
*       rather than orthogonal polynomials
*************************************************************/
double * 
legendre_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep)
{

    double * real_roots = NULL; // output
    *nkeep = 0;

    double m = (p->upper_bound - p->lower_bound) / 
            (p->p->upper - p->p->lower);
    double off = p->upper_bound - m * p->p->upper;

    orth_poly_expansion_round(&p);
   // print_orth_poly_expansion(p,3,NULL);
    //printf("last 2 = %G\n",p->coeff[p->num_poly-1]);
    size_t N = p->num_poly-1;
    //printf("N = %zu\n",N);
    if (N == 0){
        return real_roots;
    }
    else if (N == 1){
        if (fabs(p->coeff[N]) <= ZEROTHRESH){
            return real_roots;
        }
        else{
            double root = -p->coeff[0] / p->coeff[1];
            if ( (root >= -1.0-ZEROTHRESH) && (root <= 1.0 - ZEROTHRESH)){
                if (root <-1.0){
                    root = -1.0;
                }
                else if (root > 1.0){
                    root = 1.0;
                }
                *nkeep = 1;
                real_roots = calloc_double(1);
                real_roots[0] = m*root+off;
            }
        }
    }
    else{
        /* printf("I am here\n"); */
        double * nscompanion = calloc_double(N*N); // nonstandard companion
        size_t ii;
        double hnn1 = - (double) (N) / (2.0 * (double) (N) - 1.0);
        /* double hnn1 = - 1.0 / p->p->an(N); */
        nscompanion[1] = 1.0;
        /* nscompanion[(N-1)*N] += hnn1 * p->coeff[0] / p->coeff[N]; */
        nscompanion[(N-1)*N] += hnn1 * p->coeff[0] / (p->coeff[N] * sqrt(2*N+1));
        for (ii = 1; ii < N-1; ii++){
            assert (fabs(p->p->bn(ii)) < 1e-14);
            double in = (double) ii;
            nscompanion[ii*N+ii-1] = in / ( 2.0 * in + 1.0);
            nscompanion[ii*N+ii+1] = (in + 1.0) / ( 2.0 * in + 1.0);

            /* nscompanion[(N-1)*N + ii] += hnn1 * p->coeff[ii] / p->coeff[N]; */
            nscompanion[(N-1)*N + ii] += hnn1 * p->coeff[ii] * sqrt(2*ii+1)/ p->coeff[N] / sqrt(2*N+1);
        }
        nscompanion[N*N-2] += (double) (N-1) / (2.0 * (double) (N-1) + 1.0);

        
        /* nscompanion[N*N-1] += hnn1 * p->coeff[N-1] / p->coeff[N]; */
        nscompanion[N*N-1] += hnn1 * p->coeff[N-1] * sqrt(2*(N-1)+1)/ p->coeff[N] / sqrt(2*N+1);
        
        //printf("good up to here!\n");
        //dprint2d_col(N,N,nscompanion);

        int info;
        double * scale = calloc_double(N);
        //*
        //Balance
        int ILO, IHI;
        //printf("am I here? N=%zu \n",N);
        //dprint(N*N,nscompanion);
        dgebal_("S", (int*)&N, nscompanion, (int *)&N,&ILO,&IHI,scale,&info);
        //printf("yep\n");
        if (info < 0){
            fprintf(stderr, "Calling dgebl had error in %d-th input in the legendre_expansion_real_roots function\n",info);
            exit(1);
        }

        //printf("balanced!\n");
        //dprint2d_col(N,N,nscompanion);

        //IHI = M1;
        //printf("M1=%zu\n",M1);
        //printf("ilo=%zu\n",ILO);
        //printf("IHI=%zu\n",IHI);
        //*/

        double * real = calloc_double(N);
        double * img = calloc_double(N);
        //printf("allocated eigs N = %zu\n",N);
        int lwork = 8 * (int)N;
        //printf("got lwork\n");
        double * iwork = calloc_double(8*N);
        //printf("go here");

        //dgeev_("N","N", &N, nscompanion, &N, real, img, NULL, &N,
        //        NULL, &N, iwork, &lwork, &info);
        dhseqr_("E","N",(int*)&N,&ILO,&IHI,nscompanion,(int*)&N,real,img,NULL,(int*)&N,iwork,&lwork,&info);
        //printf("done here");

        if (info < 0){
            fprintf(stderr, "Calling dhesqr had error in %d-th input in the legendre_expansion_real_roots function\n",info);
            exit(1);
        }
        else if(info > 0){
            //fprintf(stderr, "Eigenvalues are still uncovered in legendre_expansion_real_roots function\n");
           // printf("coeffs are \n");
           // dprint(p->num_poly, p->coeff);
           // printf("last 2 = %G\n",p->coeff[p->num_poly-1]);
           // exit(1);
        }

      //  printf("eigenvalues \n");
        size_t * keep = calloc_size_t(N);
        for (ii = 0; ii < N; ii++){
            /* printf("(%3.15G, %3.15G)\n",real[ii],img[ii]); */
            if ((fabs(img[ii]) < 1e-6) && (real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){
                if (real[ii] < -1.0){
                    real[ii] = -1.0;
                }
                else if (real[ii] > 1.0){
                    real[ii] = 1.0;
                }
                keep[ii] = 1;
                *nkeep = *nkeep + 1;
            }
        }
        
        
        if (*nkeep > 0){
            real_roots = calloc_double(*nkeep);
            size_t counter = 0;
            for (ii = 0; ii < N; ii++){
                if (keep[ii] == 1){
                    real_roots[counter] = real[ii]*m+off;
                    counter++;
                }
            }
        }
     

        free(keep); keep = NULL;
        free(iwork); iwork  = NULL;
        free(real); real = NULL;
        free(img); img = NULL;
        free(nscompanion); nscompanion = NULL;
        free(scale); scale = NULL;
    }

    if (*nkeep > 1){
        qsort(real_roots, *nkeep, sizeof(double), dblcompare);
    }
    return real_roots;
}

/********************************************************//**
*   Obtain the real roots of a chebyshev polynomial expansion
*
*   \param[in]     p     - orthogonal polynomial expansion
*   \param[in,out] nkeep - returns how many real roots tehre are
*
*   \return real_roots - real roots of an orthonormal polynomial expansion
*
*   \note
*       Only roots within the bounds are returned
*       Algorithm is based on eigenvalues of non-standard companion matrix from
*       Roots of Polynomials Expressed in terms of orthogonal polynomials
*       David Day and Louis Romero 2005
*
*       Multiplying by a factor of sqrt(2*N+1) because using orthonormal,
*       rather than orthogonal polynomials
*************************************************************/
double * 
chebyshev_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep)
{
    /* fprintf(stderr, "Chebyshev real_roots not finished yet\n"); */
    /* exit(1); */
    double * real_roots = NULL; // output
    *nkeep = 0;

    double m = (p->upper_bound - p->lower_bound) /  (p->p->upper - p->p->lower);
    double off = p->upper_bound - m * p->p->upper;


    /* printf("coeff pre truncate = "); dprint(p->num_, p->coeff); */
    /* for (size_t ii = 0; ii < p->num_poly; ii++){ */
    /*     if (fabs(p->coeff[ii]) < 1e-13){ */
    /*         p->coeff[ii] = 0.0; */
    /*     } */
    /* } */
    orth_poly_expansion_round(&p);
    
    size_t N = p->num_poly-1;
    if (N == 0){
        return real_roots;
    }
    else if (N == 1){
        if (fabs(p->coeff[N]) <= ZEROTHRESH){
            return real_roots;
        }
        else{
            double root = -p->coeff[0] / p->coeff[1];
            if ( (root >= -1.0-ZEROTHRESH) && (root <= 1.0 - ZEROTHRESH)){
                if (root <-1.0){
                    root = -1.0;
                }
                else if (root > 1.0){
                    root = 1.0;
                }
                *nkeep = 1;
                real_roots = calloc_double(1);
                real_roots[0] = m*root+off;
            }
        }
    }
    else{
        /* printf("I am heare\n"); */
        /* dprint(N+1, p->coeff); */
        double * nscompanion = calloc_double(N*N); // nonstandard companion
        size_t ii;

        double hnn1 = 0.5;
        double gamma = p->coeff[N];
        
        nscompanion[1] = 1.0;
        nscompanion[(N-1)*N] -= hnn1*p->coeff[0] / gamma;
        for (ii = 1; ii < N-1; ii++){
            assert (fabs(p->p->bn(ii)) < 1e-14);
            
            nscompanion[ii*N+ii-1] = 0.5; // ii-th column
            nscompanion[ii*N+ii+1] = 0.5;

            // update last column
            nscompanion[(N-1)*N + ii] -= hnn1 * p->coeff[ii] / gamma;
        }
        nscompanion[N*N-2] += 0.5;
        nscompanion[N*N-1] -= hnn1 * p->coeff[N-1] / gamma;
        
        //printf("good up to here!\n");
        /* dprint2d_col(N,N,nscompanion); */

        int info;
        double * scale = calloc_double(N);
        //*
        //Balance
        int ILO, IHI;
        //printf("am I here? N=%zu \n",N);
        //dprint(N*N,nscompanion);
        dgebal_("S", (int*)&N, nscompanion, (int *)&N,&ILO,&IHI,scale,&info);
        //printf("yep\n");
        if (info < 0){
            fprintf(stderr, "Calling dgebl had error in %d-th input in the chebyshev_expansion_real_roots function\n",info);
            exit(1);
        }

        //printf("balanced!\n");
        //dprint2d_col(N,N,nscompanion);

        //IHI = M1;
        //printf("M1=%zu\n",M1);
        //printf("ilo=%zu\n",ILO);
        //printf("IHI=%zu\n",IHI);
        //*/

        double * real = calloc_double(N);
        double * img = calloc_double(N);
        //printf("allocated eigs N = %zu\n",N);
        int lwork = 8 * (int)N;
        //printf("got lwork\n");
        double * iwork = calloc_double(8*N);
        //printf("go here");

        //dgeev_("N","N", &N, nscompanion, &N, real, img, NULL, &N,
        //        NULL, &N, iwork, &lwork, &info);
        dhseqr_("E","N",(int*)&N,&ILO,&IHI,nscompanion,(int*)&N,real,img,NULL,(int*)&N,iwork,&lwork,&info);
        //printf("done here");

        if (info < 0){
            fprintf(stderr, "Calling dhesqr had error in %d-th input in the legendre_expansion_real_roots function\n",info);
            exit(1);
        }
        else if(info > 0){
            //fprintf(stderr, "Eigenvalues are still uncovered in legendre_expansion_real_roots function\n");
           // printf("coeffs are \n");
           // dprint(p->num_poly, p->coeff);
           // printf("last 2 = %G\n",p->coeff[p->num_poly-1]);
           // exit(1);
        }

       /* printf("eigenvalues \n"); */
        size_t * keep = calloc_size_t(N);
        for (ii = 0; ii < N; ii++){
            /* printf("(%3.15G, %3.15G)\n",real[ii],img[ii]); */
            if ((fabs(img[ii]) < 1e-6) && (real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){
            /* if ((real[ii] > -1.0-1e-12) && (real[ii] < 1.0+1e-12)){                 */
                if (real[ii] < -1.0){
                    real[ii] = -1.0;
                }
                else if (real[ii] > 1.0){
                    real[ii] = 1.0;
                }
                keep[ii] = 1;
                *nkeep = *nkeep + 1;
            }
        }

        /* printf("nkeep = %zu\n", *nkeep); */
        
        if (*nkeep > 0){
            real_roots = calloc_double(*nkeep);
            size_t counter = 0;
            for (ii = 0; ii < N; ii++){
                if (keep[ii] == 1){
                    real_roots[counter] = real[ii]*m+off;
                    counter++;
                }
            }
        }
     

        free(keep); keep = NULL;
        free(iwork); iwork  = NULL;
        free(real); real = NULL;
        free(img); img = NULL;
        free(nscompanion); nscompanion = NULL;
        free(scale); scale = NULL;
    }

    if (*nkeep > 1){
        qsort(real_roots, *nkeep, sizeof(double), dblcompare);
    }
    return real_roots;
}

/********************************************************//**
*   Obtain the real roots of a orthogonal polynomial expansion
*
*   \param[in] p     - orthogonal polynomial expansion
*   \param[in] nkeep - returns how many real roots tehre are
*
*   \return real_roots - real roots of an orthonormal polynomial expansion
*
*   \note
*       Only roots within the bounds are returned
*************************************************************/
double *
orth_poly_expansion_real_roots(struct OrthPolyExpansion * p, size_t * nkeep)
{
    double * real_roots = NULL;
    enum poly_type ptype = p->p->ptype;
    switch(ptype){
    case LEGENDRE:
        real_roots = legendre_expansion_real_roots(p,nkeep);   
        break;
    case STANDARD:        
        assert (1 == 0);
        //x need to convert polynomial to standard polynomial first
        //real_roots = standard_poly_real_roots(sp,nkeep);
        break;
    case CHEBYSHEV:
        real_roots = chebyshev_expansion_real_roots(p,nkeep);
        break;
    case HERMITE:
        assert (1 == 0);
        break;
    case FOURIER:
        assert (1 == 0);
        break;
    }
    return real_roots;
}

/********************************************************//**
*   Obtain the maximum of an orthogonal polynomial expansion
*
*   \param[in] p - orthogonal polynomial expansion
*   \param[in] x - location of maximum value
*
*   \return maxval - maximum value
*   
*   \note
*       if constant function then just returns the left most point
*************************************************************/
double orth_poly_expansion_max(struct OrthPolyExpansion * p, double * x)
{
    
    double maxval;
    double tempval;

    assert(p->p->ptype != FOURIER);
    maxval = orth_poly_expansion_eval(p,p->lower_bound);
    *x = p->lower_bound;

    tempval = orth_poly_expansion_eval(p,p->upper_bound);
    if (tempval > maxval){
        maxval = tempval;
        *x = p->upper_bound;
    }
    
    if (p->num_poly > 2){
        size_t nroots;
        struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p);
        double * roots = orth_poly_expansion_real_roots(deriv,&nroots);
        if (nroots > 0){
            size_t ii;
            for (ii = 0; ii < nroots; ii++){
                tempval = orth_poly_expansion_eval(p, roots[ii]);
                if (tempval > maxval){
                    *x = roots[ii];
                    maxval = tempval;
                }
            }
        }

        free(roots); roots = NULL;
        orth_poly_expansion_free(deriv); deriv = NULL;
    }
    return maxval;
}

/********************************************************//**
*   Obtain the minimum of an orthogonal polynomial expansion
*
*   \param[in]     p - orthogonal polynomial expansion
*   \param[in,out] x - location of minimum value
*
*   \return minval - minimum value
*************************************************************/
double orth_poly_expansion_min(struct OrthPolyExpansion * p, double * x)
{
    assert(p->p->ptype != FOURIER);
    
    double minval;
    double tempval;

    minval = orth_poly_expansion_eval(p,p->lower_bound);
    *x = p->lower_bound;

    tempval = orth_poly_expansion_eval(p,p->upper_bound);
    if (tempval < minval){
        minval = tempval;
        *x = p->upper_bound;
    }
    
    if (p->num_poly > 2){
        size_t nroots;
        struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p);
        double * roots = orth_poly_expansion_real_roots(deriv,&nroots);
        if (nroots > 0){
            size_t ii;
            for (ii = 0; ii < nroots; ii++){
                tempval = orth_poly_expansion_eval(p, roots[ii]);
                if (tempval < minval){
                    *x = roots[ii];
                    minval = tempval;
                }
            }
        }
        free(roots); roots = NULL;
        orth_poly_expansion_free(deriv); deriv = NULL;
    }
    return minval;
}

/********************************************************//**
*   Obtain the maximum in absolute value of an orthogonal polynomial expansion
*
*   \param[in]     p     - orthogonal polynomial expansion
*   \param[in,out] x     - location of maximum
*   \param[in]     oargs - optimization arguments 
*                          required for HERMITE otherwise can set NULL
*
*   \return maxval : maximum value (absolute value)
*
*   \note
*       if no roots then either lower or upper bound
*************************************************************/
double orth_poly_expansion_absmax(
    struct OrthPolyExpansion * p, double * x, void * oargs)
{

    //printf("in absmax\n");
   // print_orth_poly_expansion(p,3,NULL);
    //printf("%G\n", orth_poly_expansion_norm(p));

    enum poly_type ptype = p->p->ptype;
    if (oargs != NULL){

        struct c3Vector * optnodes = oargs;
        double mval = fabs(orth_poly_expansion_eval(p,optnodes->elem[0]));
        *x = optnodes->elem[0];
        double cval = mval;
        if (ptype == HERMITE){
            mval *= exp(-pow(optnodes->elem[0],2)/2.0);
        }
        *x = optnodes->elem[0];
        for (size_t ii = 0; ii < optnodes->size; ii++){
            double val = fabs(orth_poly_expansion_eval(p,optnodes->elem[ii]));
            double tval = val;
            if (ptype == HERMITE){
                val *= exp(-pow(optnodes->elem[ii],2)/2.0);
                //printf("ii=%zu, x = %G. val=%G, tval=%G\n",ii,optnodes->elem[ii],val,tval);
            }
            if (val > mval){
//                printf("min achieved\n");
                mval = val;
                cval = tval;
                *x = optnodes->elem[ii];
            }
        }
//        printf("optloc=%G .... cval = %G\n",*x,cval);
        return cval;
    }
    else if ((ptype == HERMITE) || (ptype == FOURIER)){
        fprintf(stderr,"Must specify optimizatino arguments\n");
        fprintf(stderr,"In the form of candidate points for \n");
        fprintf(stderr,"finding the absmax of hermite or fourier expansion\n");
        exit(1);
        
    }
    double maxval;
    double norm = orth_poly_expansion_norm(p);
    
    if (norm < ZEROTHRESH) {
        *x = p->lower_bound;
        maxval = 0.0;
    }
    else{
        //printf("nroots=%zu\n", nroots);
        double tempval;

        maxval = fabs(orth_poly_expansion_eval(p,p->lower_bound));
        *x = p->lower_bound;

        tempval = fabs(orth_poly_expansion_eval(p,p->upper_bound));
        if (tempval > maxval){
            maxval = tempval;
            *x = p->upper_bound;
        }
        if (p->num_poly > 2){
            size_t nroots;
            struct OrthPolyExpansion * deriv = orth_poly_expansion_deriv(p);
            double * roots = orth_poly_expansion_real_roots(deriv,&nroots);
            if (nroots > 0){
                size_t ii;
                for (ii = 0; ii < nroots; ii++){
                    tempval = fabs(orth_poly_expansion_eval(p, roots[ii]));
                    if (tempval > maxval){
                        *x = roots[ii];
                        maxval = tempval;
                    }
                }
            }

            free(roots); roots = NULL;
            orth_poly_expansion_free(deriv); deriv = NULL;
        }
    }
    //printf("done\n");
    return maxval;
}


/////////////////////////////////////////////////////////
// Utilities
char * convert_ptype_to_char(enum poly_type ptype)
{   
    char * out = NULL;
    switch (ptype) {
        case LEGENDRE:
            out = "Legendre";
            break;
        case HERMITE:
            out = "Hermite";
            break;
        case CHEBYSHEV:
            out = "Chebyshev";
            break;
        case STANDARD:
            out =  "Standard";
            break;
        case FOURIER:
            out =  "Fourier";
            break;            
        //default:
        //    fprintf(stderr, "Polynomial type does not exist: %d\n ", ptype);
    }
    return out;

}
void print_orth_poly_expansion(struct OrthPolyExpansion * p, size_t prec, 
                               void * args, FILE *fp)
{

    if (args == NULL){
        fprintf(fp, "Orthogonal Polynomial Expansion:\n");
        fprintf(fp, "--------------------------------\n");
        fprintf(fp, "Polynomial basis is %s\n",convert_ptype_to_char(p->p->ptype));
        fprintf(fp, "Coefficients = ");
        size_t ii;
        if (p->p->ptype == FOURIER){
            for (ii = 0; ii < p->num_poly; ii++){
                fprintf(fp, "%3.3f %3.3f\n",
                        creal(p->ccoeff[ii]), cimag(p->ccoeff[ii]));
            }            
        }
        else{
            for (ii = 0; ii < p->num_poly; ii++){
                if (prec == 0){
                    fprintf(fp, "%3.1f ", p->coeff[ii]);
                }
                else if (prec == 1){
                    fprintf(fp, "%3.3f ", p->coeff[ii]);
                }
                else if (prec == 2){
                    fprintf(fp, "%3.15f ", p->coeff[ii]);
                }
                else{
                    fprintf(fp, "%3.15E ", p->coeff[ii]);
                }
            }
        }
        printf("\n");
    }
}

