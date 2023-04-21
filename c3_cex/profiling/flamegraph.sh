#!/bin/bash -ex
perf script | ~/Software/FlameGraph/stackcollapse-perf.pl > out.perf-folded
~/Software/FlameGraph/flamegraph.pl --title=flamegraph out.perf-folded > flamegraph.svg
~/Software/FlameGraph/flamegraph.pl --title=flamegraph out.perf-folded --reverse > flamegraph_rev.svg
