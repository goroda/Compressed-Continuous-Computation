C3HOME=~/Documents/c3


#SAMPLING 
NSAMPLE=100
RVTYPE=uniform
DIM=2
LB=0.0
UB=1.0
FILENAME="trainingx.dat"

GENSAMPLES="$C3HOME/bin/random_sample -r $NSAMPLE -t $RVTYPE -c $DIM -l $LB -u $UB"

echo $GENSAMPLES
$GENSAMPLES > $FILENAME

#Function Evaluation
FUNC=2
EVALFILE="trainingy.dat"

EVALFUNC="$C3HOME/bin/simlib_util -f $FUNC -i $FILENAME -n $NSAMPLE -o $EVALFILE"

echo $EVALFUNC
$EVALFUNC


#Regress

REGRESS="$C3HOME/profiling/regress/bin/aioregress -x $FILENAME -y $EVALFILE"

echo $REGRESS
# $REGRESS # Just generate regression


# Do profiling
# valgrind --tool=callgrind $REGRESS
# python $C3HOME/profiling/gprof2dot.py -f callgrind callgrind.out.* | dot -Tsvg -o output.svg
# rm callgrind.out.*
