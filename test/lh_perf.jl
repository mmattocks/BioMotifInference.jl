using BenchmarkTools, BioMotifInference, BioBackgroundModels, BioSequences
println(Threads.nthreads())
source_pwm = [.7 .1 .1 .1
.1 .1 .1 .7
.1 .1 .7 .1]
source_pwm_2 = [.6 .1 .1 .2
.2 .1 .1 .6
.1 .2 .6 .1]
sources = [(log.(source_pwm), 1),(log.(source_pwm_2), 1)]

bg_scores = log.(fill(.1, (150,10000)))

obs=zeros(UInt8,151,10000)
obs[1:150,1:end].=1
obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]
mix=trues(10000,2)

println(median(@benchmark (BioMotifInference.IPM_likelihood($sources, $obs, $obsl, $bg_scores, $mix)) samples=5 evals=20))

println(median(@benchmark (BioMotifInference.dev_likelihood($sources, $obs, $obsl, $bg_scores, $BitMatrix(transpose(mix)))) samples=5 evals=20))