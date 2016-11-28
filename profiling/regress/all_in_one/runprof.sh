C3HOME=~/Documents/c3

######################################
# Function Information
######################################
RVTYPE=uniform
DIM=2
LB=0.0
UB=1.0
FUNC=2

######################################
# Generate Training Data
######################################

# Sample training data
NSAMPLE=100
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

FTFILE="trainedft.c3"
REGRESS="$C3HOME/profiling/regress/bin/aioregress -x $FILENAME -y $EVALFILE -v 1 -o $FTFILE"

echo $REGRESS
$REGRESS # Just generate regression

# Do profiling
# valgrind --tool=callgrind $REGRESS
# python $C3HOME/profiling/gprof2dot.py -f callgrind callgrind.out.* | dot -Tsvg -o output.svg
# rm callgrind.out.*
