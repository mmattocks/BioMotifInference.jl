set memory $JULIA_NUM_THREADS
set -U -x JULIA_NUM_THREADS 1
julia lh_perf.jl
set -U -x JULIA_NUM_THREADS 2
julia lh_perf.jl
set -U -x JULIA_NUM_THREADS 4
julia lh_perf.jl
set -U -x JULIA_NUM_THREADS 6
julia lh_perf.jl
set -U -x JULIA_NUM_THREADS 8
julia lh_perf.jl
set -U -x JULIA_NUM_THREADS 12
julia lh_perf.jl
set -U -x JULIA_NUM_THREADS $memory
echo "done"