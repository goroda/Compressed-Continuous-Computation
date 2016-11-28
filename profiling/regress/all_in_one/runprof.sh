C3HOME=~/Documents/c3

######################################
# Function Information
######################################
RVTYPE=uniform
DIM=2
LB=-2.0
UB=2.0
FUNC=1

######################################
# Generate Training Data
######################################

# Sample training data
NSAMPLE=70
FILENAME="trainingx.dat"
GENSAMPLES="$C3HOME/bin/random_sample -r $NSAMPLE -t $RVTYPE -c $DIM -l $LB -u $UB"

echo $GENSAMPLES
$GENSAMPLES > $FILENAME

# Function Evaluation
EVALFILE="trainingy.dat"
EVALFUNC="$C3HOME/bin/simlib_util -f $FUNC -i $FILENAME -n $NSAMPLE -o $EVALFILE"

echo $EVALFUNC
$EVALFUNC

######################################
# Perform Regression
######################################
VERBOSE=2
FTFILE="trainedft.c3"
REGRESS="$C3HOME/profiling/regress/bin/aioregress -x $FILENAME -y $EVALFILE -v $VERBOSE -o $FTFILE"

echo $REGRESS
$REGRESS # Just generate regression

# Do profiling
# valgrind --tool=callgrind $REGRESS
# python $C3HOME/profiling/gprof2dot.py -f callgrind callgrind.out.* | dot -Tsvg -o output.svg
# rm callgrind.out.*

######################################
# Generate Testing Data
######################################
NSAMPLE_TEST=10000
XFILE_TEST="test_x.dat"
GENTESTSAMPLES="$C3HOME/bin/random_sample -r $NSAMPLE_TEST -t $RVTYPE -c $DIM -l $LB -u $UB"

echo $GENTESTSAMPLES
$GENTESTSAMPLES > $XFILE_TEST

######################################
# Evaluate the true function
######################################
YFILE_TEST="test_y.dat"
TESTVALSFUNC="$C3HOME/bin/simlib_util -f $FUNC -i $XFILE_TEST -n $NSAMPLE_TEST -o $YFILE_TEST"
echo $TESTVALSFUNC
$TESTVALSFUNC

######################################
# Evaluate the regressed function
######################################
EVALFT="$C3HOME/bin/ftstats -x $XFILE_TEST -i $FTFILE"
echo $EVALFT
$EVALFT > ftevals.dat
