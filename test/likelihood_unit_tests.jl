@testset "Model scoring and likelihood functions" begin
    #test a trivial scoring example
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("AAAAA")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))

    source_pwm = zeros(1,4)
    source_pwm[1,1] = 1
    log_pwm = log.(source_pwm)

    source_stop=5

    #score prellocates
    ss_srcs = [revcomp_pwm(log_pwm)]
    ds_srcs=[cat(log_pwm,revcomp_pwm(log_pwm),dims=3)]
    score_mat_ds=zeros(source_stop,2)
    score_mat_ss=zeros(source_stop,1)
    score_matrices_ds=Vector{Matrix{Float64}}(undef, 2)
    score_matrices_ss=Vector{Vector{Float64}}(undef, 2)

    score_sources_ds!(score_mat_ds, score_matrices_ds, ds_srcs, obs[:,1], source_stop)
    @test score_matrices_ds[1] == [0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf]

    score_sources_ss!(score_mat_ss, score_matrices_ss, ss_srcs, obs[:,1], source_stop)

    @test score_matrices_ss[1] == [-Inf for i in 1:5]

    #test a more complicated scoring example with two sources
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATGATGATG")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))

    source_pwm = [.7 .1 .1 .1
                  .1 .1 .1 .7
                  .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    target_s1 = [.7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3]
    target_s2 = [.6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2)]
    log_pwms = [log.(source_pwm), log.(source_pwm_2)]

    source_start = 1
    source_stop = 10

    #score prellocates
    ss_srcs = log_pwms
    ds_srcs=[cat(pwm,revcomp_pwm(pwm),dims=3) for pwm in log_pwms]
    score_mat_ds=zeros(source_stop,2)
    score_mat_ss=zeros(source_stop,1)

    score_sources_ds!(score_mat_ds, score_matrices_ds, ds_srcs, obs[:,1], [source_stop, source_stop])
    @test isapprox(score_matrices_ds[1], log.(target_s1))
    @test isapprox(score_matrices_ds[2], log.(target_s2))

    score_sources_ss!(score_mat_ss, score_matrices_ss, ss_srcs, obs[:,1], [source_stop, source_stop])
    @test isapprox(score_matrices_ss[1], log.(target_s1[:,1]))
    @test isapprox(score_matrices_ss[2], log.(target_s2[:,1]))

    #test score weaving and IPM likelihood calculations
    #trivial example using fake cardinality_penalty
    target=logaddexp(log(.25), 0.)
    lh_vec=zeros(2)
    obsl=1
    bg_scores=view([log(.25)],:,:)
    score_mat=[[0.]]
    osi=[1]
    source_wmls=[1]
    lme=0.
    cardinality_penalty=0. #not correct but for test purposes
    osi_emit=Vector{Int64}()

    @test weave_scores_ss!(lh_vec, obsl, bg_scores, score_mat, osi, source_wmls, lme, cardinality_penalty, osi_emit) == target

    target2=logsumexp([log(.25), log(.5), log(.5)])
    lh_vec=zeros(2)
    weavevec=zeros(3)
    score_mat=[zeros(1,2)]
    empty!(osi_emit)
    lme=log(.5)

    @test weave_scores_ds!(weavevec, lh_vec, obsl, bg_scores, score_mat, osi, source_wmls, lme, cardinality_penalty, osi_emit) == target2

    #test more complicated weaves
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATGATGATG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGATGATGA")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    
    sources = [(log.(source_pwm), 1),(log.(source_pwm_2), 1)]
    dbl1_sources=[(log.(source_pwm), 1),(log.(source_pwm), 1)]

    ds_srcs=[cat(pwm[1],revcomp_pwm(pwm[1]),dims=3) for pwm in sources]
    dbl1_srcs=[cat(pwm[1],revcomp_pwm(pwm[1]),dims=3) for pwm in dbl1_sources]

    source_stops=[10,10]

    target_o2_s1 = [.1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3]
    target_o2_s2 = [(.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2)]


    score_mat_ds=zeros(source_stop,2)
    score_matrices_ds=Vector{Matrix{Float64}}(undef, 2)
    
    score_sources_ds!(score_mat_ds, score_matrices_ds, ds_srcs, obs[:,1], source_stops)

    o1_mats = copy(score_matrices_ds)

    @test isapprox(o1_mats, [log.(target_s1), log.(target_s2)])

    score_sources_ds!(score_mat_ds, score_matrices_ds, ds_srcs, obs[:,2], source_stops)

    o2_mats = copy(score_matrices_ds)

    @test isapprox(o2_mats, [log.(target_o2_s1), log.(target_o2_s2)])

    bg_scores = log.(fill(.5, (12,2)))
    log_motif_expectation = log(0.5 / size(bg_scores,1))
    osi = [1,2]
    obs_cardinality = length(osi)
    penalty_sum = sum(exp.(fill(log_motif_expectation,obs_cardinality)))
    cardinality_penalty=log(1.0-penalty_sum)
    empty!(osi_emit)
    source_wmls=[3,3]

    lh_target = -8.87035766177774
    lh_vec=zeros(13)

    o1_lh = weave_scores_ds!(weavevec, lh_vec, 12, view(bg_scores,:,1), o1_mats, osi, source_wmls, log_motif_expectation, cardinality_penalty, osi_emit)
    @test isapprox(lh_target,o1_lh)

    empty!(osi_emit)

    o2_lh = weave_scores_ds!(weavevec, lh_vec, 12, view(bg_scores,:,2), o2_mats, osi, source_wmls, log_motif_expectation, cardinality_penalty, osi_emit)

    lh,cache = IPM_likelihood(sources, obs, [12,12], bg_scores, trues(2,2), true,true)
    @test isapprox(o1_lh,cache[1])
    @test isapprox(o2_lh,cache[2])
    @test isapprox(lps(o1_lh,o2_lh),lh)
    
    #test source penalization
    score_sources_ds!(score_mat_ds, score_matrices_ds, dbl1_srcs, obs[:,1], source_stops)

    dbl_score_mat=copy(score_matrices_ds)

    empty!(osi_emit)

    naive=weave_scores_ds!(weavevec, lh_vec, 12, view(bg_scores,:,1),Vector{Matrix{Float64}}(), Vector{Int64}(), Vector{Int64}(), log_motif_expectation, cardinality_penalty, osi_emit)

    empty!(osi_emit)

    single=weave_scores_ds!(weavevec, lh_vec, 12, view(bg_scores,:,1), [dbl_score_mat[1]], [1], [3], log_motif_expectation, cardinality_penalty, osi_emit)

    empty!(osi_emit)

    double=weave_scores_ds!(weavevec, lh_vec, 12, view(bg_scores,:,1), dbl_score_mat, [1,2], [3,3], log_motif_expectation, cardinality_penalty, osi_emit)

    empty!(osi_emit)

    triple=weave_scores_ds!(weavevec, lh_vec, 12, view(bg_scores,:,1), [dbl_score_mat[1] for i in 1:3], [1,2,3], [3,3,3], log_motif_expectation, cardinality_penalty, osi_emit)

    @test (single-naive) > (double-single) > (triple-double)

    naive_target=-16.635532333438686
    naive = IPM_likelihood(sources, obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores,falses(size(obs)[2],length(sources)))
    @test naive==naive_target

    #test IPM_likelihood clean vector and cache calculations
    mix_matrix=trues(2,2)
    baselh,basecache = IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true,true)

    clean=[true,false]

    unchangedlh,unchangedcache=IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true, true, basecache, clean)

    changed_lh,changedcache=IPM_likelihood(sources, obs, [12,12], bg_scores, BitMatrix([true true; false false]), true, true, basecache, clean)

    indep_lh, indepcache=IPM_likelihood(sources, obs,[12,12], bg_scores, BitMatrix([true true; false false]), true, true)

    @test baselh==unchangedlh!=changed_lh==indep_lh
    @test basecache==unchangedcache!=changedcache==indepcache

    #check that IPM_likelihood works in single stranded operation
    IPM_likelihood(sources, obs,[12,12], bg_scores, BitMatrix([true true; false false]), false)
end
