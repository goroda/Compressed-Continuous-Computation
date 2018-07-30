#!/usr/bin/gnuplot
#

reset

# wxt
#set terminal wxt size 1024,343 enhanced font 'Verdana,10' persist
# png
set terminal pngcairo size 1000,1000 enhanced font 'Verdana,10'
set output 'input_data.png'

unset border
unset key
# unset tics
unset colorbox

load '/home/goroda/Software/gnuplot/gnuplot-colorbrewer/qualitative/Dark2.plt'
# load '/home/goroda/Software/gnuplot/gnuplot-colorbrewer/qualitative/Accent.plt'

traj = 'lorenz_init_010_dt_001.dat'

set multiplot layout 4,1 # columnsfirst margins 0.2,0.9,.1,.9 spacing 0,0
set ylabel "x"
plot traj u 1:2 w l ls 1 notitle

unset xlabel
set ylabel "y"
plot traj u 1:3 w l ls 2 notitle

set ylabel "z"
plot traj u 1:4 w l ls 3 notitle

set ylabel "dist"
plot traj u 1:5 w l ls 4 notitle


###############################################################
###############################################################
###############################################################
###############################################################
## Results
###############################################################
###############################################################
###############################################################
###############################################################

unset multiplot

reset

set output 'output_data.png'
traj = 'out.dat'

unset border
unset key
# unset tics
unset colorbox

set multiplot layout 3,1 # columnsfirst margins 0.2,0.9,.1,.9 spacing 0,0
set ylabel "dx"
plot traj u 1:($2-$3) w l ls 1 notitle

unset xlabel
set ylabel "dy"
plot traj u 1:($4-$5) w l ls 2 notitle

set ylabel "dz"
plot traj u 1:($6-$7) w l ls 3 notitle


###############################################################
## Results zoomed
###############################################################

unset multiplot

reset

set output 'output_data_zoomed.png'
traj = 'out.dat'

set yrange [-1:1]
unset border
unset key
# unset tics
unset colorbox

set multiplot layout 3,1 # columnsfirst margins 0.2,0.9,.1,.9 spacing 0,0
set ylabel "dx"
plot traj u 1:($2-$3) w l ls 1 notitle

unset xlabel
set ylabel "dy"
plot traj u 1:($4-$5) w l ls 2 notitle

set ylabel "dz"
plot traj u 1:($6-$7) w l ls 3 notitle
