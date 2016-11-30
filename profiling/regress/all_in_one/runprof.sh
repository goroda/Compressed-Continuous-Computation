#C3HOME=~/Documents/c3
export C3HOME=~/Software/c3

######################################
# Function Information
######################################
RVTYPE=uniform
DIM=2
LB=0.0
UB=1.0
FUNC=2

######################################
# Regression options (order=5,rank=4 for profile data)
######################################
MAXORDER=4
RANK=2
VERBOSE=1
FTFILE="trainedft.c3"

######################################
# Generate Training Data (1000 for profile data)
######################################

# Sample training data
NSAMPLE=1000
FILENAME="trainingx.dat"
GENSAMPLES="$C3HOME/bin/random_sample -r $NSAMPLE -t $RVTYPE -c $DIM -l $LB -u $UB"

echo "$GENSAMPLES > $FILENAME"
$GENSAMPLES > $FILENAME

# Function Evaluation
EVALFILE="trainingy.dat"
EVALFUNC="$C3HOME/bin/simlib_util -f $FUNC -i $FILENAME -n $NSAMPLE -o $EVALFILE"

echo $EVALFUNC
$EVALFUNC

######################################
# Perform Regression
######################################
REGRESS="$C3HOME/profiling/regress/bin/aioregress -x $FILENAME -y $EVALFILE -m $MAXORDER -r $RANK -v $VERBOSE -o $FTFILE"
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

echo "$GENTESTSAMPLES > $XFILE_TEST"
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

######################################
# Get Squared Error
######################################
error=$(awk 'FNR==NR { file1[NR]=$1; next; }; { diff=$1-file1[FNR]; sum+=diff^2;}; 
  END { print sum/FNR; }' test_y.dat ftevals.dat)
echo "Relative Error: $error"



