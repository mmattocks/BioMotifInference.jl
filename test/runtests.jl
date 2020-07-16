@info "Loading test packages..."

using BioMotifInference, BioBackgroundModels, BioSequences, Distributions, Distributed, Random, Serialization, Test
import StatsFuns: logsumexp
import BioMotifInference:estimate_dirichlet_prior_on_wm, assemble_source_priors, init_logPWM_sources, wm_shift, permute_source_weights, get_length_params, permute_source_length, get_pwm_info, get_erosion_idxs, erode_source, init_mix_matrix, mixvec_decorrelate, mix_matrix_decorrelate, most_dissimilar, most_similar, revcomp_pwm, score_source, score_obs_sources, weave_scores, IPM_likelihood, consolidate_check, consolidate_srcs, pwm_distance, permute_source, permute_mix, perm_src_fit_mix, fit_mix, random_decorrelate, reinit_src, erode_model, reinit_src, distance_merge, similarity_merge, converge_ensemble!, reset_ensemble, PRIOR_WT
import Distances: euclidean

@info "Beginning tests..."

Random.seed!(786)
O=1000;S=50

@testset "PWM source prior setup, PWM source initialisation and manipulation functions" begin
    #test dirichlet prior estimation from wm inputs
    wm_input = [.0 .2 .3 .5; .0 .2 .3 .5]
    est_dirichlet_vec = estimate_dirichlet_prior_on_wm(wm_input)
    @test typeof(est_dirichlet_vec) == Vector{Dirichlet{Float64}}
    for pos in 1:length(est_dirichlet_vec)
        @test isapprox(est_dirichlet_vec[pos].alpha, wm_input[pos,:].*PRIOR_WT)
    end

    bad_input = wm_input .* 2
    @test_throws DomainError estimate_dirichlet_prior_on_wm(bad_input)

    wm_input = [.1 .2 .3 .4; .1 .2 .3 .4]
    est_dirichlet_vec = estimate_dirichlet_prior_on_wm(wm_input)
    @test typeof(est_dirichlet_vec) == Vector{Dirichlet{Float64}}
    for pos in 1:length(est_dirichlet_vec)
        @test est_dirichlet_vec[pos].alpha == wm_input[pos,:].*PRIOR_WT
    end

    length_range = 2:2

    #test informative/uninformative source prior vector assembly
    test_priors = assemble_source_priors(2, [wm_input])
    @test length(test_priors)  == 2
    for pos in 1:length(test_priors[1])
        @test test_priors[1][pos].alpha == wm_input[pos,:].*PRIOR_WT
    end
    @test test_priors[2] == false

    #test source wm initialisation from priors
    test_sources = init_logPWM_sources(test_priors, length_range)
    for source in test_sources
        for pos in 1:size(source[1])[1]
            @test isprobvec(exp.(source[1][pos,:]))
        end
    end

    #test that wm_shift is returning good shifted probvecs
    rando_dist=Dirichlet([.25,.25,.25,.25])
    for i in 1:1000
        wm=rand(rando_dist)
        new_wm=wm_shift(wm,Weibull(1.5,.1))
        @test isprobvec(new_wm)
        @test wm!=new_wm
    end

    #test that legal new sources are generated by permute_source_weights
    permuted_weight_sources=deepcopy(test_sources)
    permuted_weight_sources[1]=permute_source_weights(permuted_weight_sources[1],1.,Weibull(1.5,.1))
    permuted_weight_sources[2]=permute_source_weights(permuted_weight_sources[2],1.,Weibull(1.5,.1))
    @test permuted_weight_sources != test_sources
    for (s,source) in enumerate(permuted_weight_sources)
        for pos in 1:size(source[1],1)
            @test isprobvec(exp.(source[1][pos,:]))
            @test source[1][pos,:] != test_sources[s][1][pos,:]
        end
    end

    #test that get_length_params returns legal length shifts
    lls=1:10
    pr=1:5
    for i in 1:1000
        srcl=rand(1:10)
        sign, permute_length=get_length_params(srcl, lls, pr)
        @test pr[1]<=permute_length<=pr[end]
        @test sign==-1 || sign==1
        @test lls[1]<=(srcl+(sign*permute_length))<=lls[end]
    end

    permuted_length_sources=deepcopy(test_sources)
    permuted_length_sources[1]=permute_source_length(permuted_length_sources[1],test_priors[1],1:3,1:10)
    permuted_length_sources[2]=permute_source_length(permuted_length_sources[2],test_priors[2],1:3,1:10)
    for (s, source) in enumerate(permuted_length_sources)
        @test size(source[1],1) != size(test_sources[1][1],1)
        @test 1<=size(source[1],1)<=3
    end

    info_test_wm=[1. 0. 0. 0.
    .94 .02 .02 .02
    .82 .06 .06 .06
    .7 .1 .1 .1
    .67 .11 .11 .11
    .52 .16 .16 .16
    .4 .2 .2 .2
    .25 .25 .25 .25]

    #test eroding sources by finding most informational position and cutting off when information drops below threshold
    infovec=get_pwm_info(log.(info_test_wm))
    @test infovec==[2.0, 1.5774573308022544, 1.0346297041419121, 0.6432203505529603, 0.5620360019822908, 0.2403724636586433, 0.07807190511263773, 0.0]

    erosion_test_source=(log.([.25 .25 .25 .25
                         .2 .4 .2 .2
                         .7 .1 .1 .1
                         .06 .06 .06 .82
                         .7 .1 .1 .1
                         .25 .25 .25 .25]),1)

    infovec=get_pwm_info(erosion_test_source[1])
    start_idx, end_idx = get_erosion_idxs(infovec, .25, 2:8)
    @test start_idx==3
    @test end_idx==5

    eroded_pwm,eroded_prior_idx=erode_source(erosion_test_source,2:8,.25)
    for pos in 1:size(eroded_pwm,1)
        @test isprobvec(exp.(eroded_pwm[pos,:]))
    end
    @test eroded_prior_idx==3
    @test isapprox(exp.(eroded_pwm),[.7 .1 .1 .1
    .06 .06 .06 .82
    .7 .1 .1 .1])
end

@testset "Mix matrix initialisation and manipulation functions" begin
    #test mix matrix init
    prior_mix_test=init_mix_matrix((trues(2,10),0.0),2, 20)
    @test all(prior_mix_test[:,1:10])
    @test !any(prior_mix_test[:,11:20])

    @test sum(init_mix_matrix((falses(0,0),1.0), O, S)) == O*S
    @test sum(init_mix_matrix((falses(0,0),0.0), O, S)) == 0
    @test 0 < sum(init_mix_matrix((falses(0,0),0.5), O, S)) < O*S

    #test mix matrix decorrelation
    empty_mixvec=falses(O)
    one_mix=mixvec_decorrelate(empty_mixvec,1)
    @test sum(one_mix)==1

    empty_mix = falses(O,S)
    new_mix,clean=mix_matrix_decorrelate(empty_mix, 500)
    @test 0 < sum(new_mix) <= 500
    @test !all(clean)

    full_mix = trues(O,S)
    less_full_mix,clean=mix_matrix_decorrelate(full_mix, 500)

    @test O*S-sum(less_full_mix) <= 500
    @test !all(clean)

    #test matrix similarity and dissimilarity functions
    test_mix=falses(O,S)
    test_mix[1:Int(floor(O/2)),:].=true
    test_idx=3
    compare_mix=deepcopy(test_mix)
    compare_mix[:,test_idx] .= .!compare_mix[:,test_idx]
    @test most_dissimilar(test_mix,compare_mix)==test_idx

    src_mixvec=falses(O)
    src_mixvec[Int(ceil(O/2)):end].=true
    @test most_similar(src_mixvec,compare_mix)==test_idx
end

@testset "Observation setup and model scoring functions" begin
    #test a trivial scoring example
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("AAAAA")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))

    source_pwm = zeros(1,4)
    source_pwm[1,1] = 1
    log_pwm = log.(source_pwm)

    source_stop=5

    @test score_source(obs[1:5,1], log_pwm, source_stop) == [0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf; 0.0 -Inf]

    #make sure revcomp_pwm is reversing pwms across both dimensions
    revcomp_test_pwm = zeros(2,4)
    revcomp_test_pwm[1,1] = 1
    revcomp_test_pwm[2,3] = 1
    log_revcomp_test_pwm = log.(revcomp_test_pwm)
    @test revcomp_pwm(log_revcomp_test_pwm) == [-Inf 0 -Inf -Inf
                                                        -Inf -Inf -Inf 0]

    #test a more complicated scoring example
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATGATGATG")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))

    source_pwm = [.7 .1 .1 .1
                  .1 .1 .1 .7
                  .1 .1 .7 .1]
    log_pwm = log.(source_pwm)

    source_start = 1
    source_stop = 10
    
    @test isapprox(exp.(score_source(obs[1:12,1], log_pwm, source_stop)),
        [.7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3])

    #test scoring of multiple obs and sources
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATGATGATG")
         BioSequences.LongSequence{DNAAlphabet{2}}("TGATGATGATGA")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    
    source_pwm_2 = [.6 .1 .1 .2
                    .2 .1 .1 .6
                    .1 .2 .6 .1]

    target_o1_s1 = [.7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3]
    target_o1_s2 = [.6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2)]
    target_o2_s1 = [.1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3; .1^3 (.1*.7^2); .7^3 .1^3; .1^3 .1^3]
    target_o2_s2 = [(.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2); (.2*.1^2) (.2*.6^2); .6^3 (.2*.1^2); (.2*.1^2) (.2*.1^2)]

    sources = [(log.(source_pwm), 1),(log.(source_pwm_2), 1)]
    dbl1_srcs=[(log.(source_pwm), 1),(log.(source_pwm), 1)]
    source_wmls = [size(source[1])[1] for source in sources]

    position_start = 1
    offsets=[0,0]
    mix_matrix = trues(2,2)

    score_mat1 = score_obs_sources(sources, Vector(obs[:,1]), 12, source_wmls)

    @test isapprox(exp.(score_mat1[1]),target_o1_s1)
    @test isapprox(exp.(score_mat1[2]),target_o1_s2)

    score_mat2 = score_obs_sources(sources, Vector(obs[:,2]), 12, source_wmls)

    @test isapprox(exp.(score_mat2[1]),target_o2_s1)
    @test isapprox(exp.(score_mat2[2]),target_o2_s2)

    #test score weaving and IPM likelihood calculations
    o=1
    bg_scores = log.(fill(.5, (12,2)))
    log_motif_expectation = log(0.5 / size(bg_scores)[1])
    obs_source_indices = findall(mix_matrix[o,:])
    obs_cardinality = length(obs_source_indices)
    penalty_sum = sum(exp.(fill(log_motif_expectation,obs_cardinality)))
    cardinality_penalty=log(1.0-penalty_sum) 

    lh_target = -8.87035766177774

    o1_lh = weave_scores(12, view(bg_scores,:,1), score_mat1, findall(mix_matrix[1,:]), source_wmls, log_motif_expectation, cardinality_penalty)
    @test isapprox(lh_target,o1_lh)
    o2_lh = weave_scores(12, view(bg_scores,:,2), score_mat2, findall(mix_matrix[2,:]), source_wmls, log_motif_expectation, cardinality_penalty)

    lh,cache = IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true,true)
    @test o1_lh==cache[1]
    @test o2_lh==cache[2]
    @test isapprox(lps(o1_lh,o2_lh),lh)
    
    #test source penalization
    dbl_score_mat= score_obs_sources(dbl1_srcs, Vector(obs[:,1]),12,source_wmls)

    naive=weave_scores(12, view(bg_scores,:,1), Vector{Matrix{Float64}}(), Vector{Int64}(), Vector{Int64}(), log_motif_expectation,cardinality_penalty)

    single=weave_scores(12, view(bg_scores,:,1), [dbl_score_mat[1]], [1], [source_wmls[1]], log_motif_expectation, cardinality_penalty)

    double=weave_scores(12, view(bg_scores,:,1), dbl_score_mat, [1,2], source_wmls, log_motif_expectation, cardinality_penalty)

    triple=weave_scores(12, view(bg_scores,:,1), [dbl_score_mat[1] for i in 1:3], [1,2,3], [3 for i in 1:3], log_motif_expectation, cardinality_penalty)

    @test (single-naive) > (double-single) > (triple-double)

    naive_target=-16.635532333438686
    naive = IPM_likelihood(sources, obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores,falses(size(obs)[2],length(sources)))
    @test naive==naive_target

    #test IPM_likelihood clean vector and cache calculations
    baselh,basecache = IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true,true)

    clean=[true,false]

    unchangedlh,unchangedcache=IPM_likelihood(sources, obs, [12,12], bg_scores, mix_matrix, true, true, basecache, clean)

    changed_lh,changedcache=IPM_likelihood(sources, obs, [12,12], bg_scores, BitMatrix([true true; false false]), true, true, basecache, clean)

    indep_lh, indepcache=IPM_likelihood(sources,obs,[12,12], bg_scores, BitMatrix([true true; false false]), true, true)

    @test baselh==unchangedlh!=changed_lh==indep_lh
    @test basecache==unchangedcache!=changedcache==indepcache
end

@testset "Orthogonality helper" begin
    bg_scores = log.(fill(.25, (17,3)))
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATTACGATGATGCA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCAGTTACGATGATCAG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TTACGCACAGATGTTAC")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    src_ATG = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    src_CAG = [.1 .7 .1 .1
    .7 .1 .1 .1
    .1 .1 .7 .1]

    src_TTAC = [.1 .1 .1 .7
    .1 .1 .1 .7
    .7 .1 .1 .1
    .1 .7 .1 .1]

    src_GCA = [.1 .1 .7 .1
    .1 .7 .1 .1
    .7 .1 .1 .1]

    consolidate_one = [(log.(src_ATG),0),(log.(src_TTAC),0),(log.(src_ATG),0)]
    cons_one_mix = BitMatrix([true true false
                    true false true
                    false true true])

    consolidate_two = [(log.(src_ATG),0),(log.(src_ATG),0),(log.(src_ATG),0)]
    cons_two_mix = BitMatrix([true false false
                    false true false
                    false false true])

    distance_model = [(log.(src_TTAC),0),(log.(src_CAG),0),(log.(src_GCA),0)]

    c1model = ICA_PWM_Model("c1", consolidate_one, 3, 3:4, cons_one_mix, IPM_likelihood(consolidate_one, obs, obsl, bg_scores, cons_one_mix),[""])
    c2model = ICA_PWM_Model("c2", consolidate_two, 3, 3:4, cons_two_mix, IPM_likelihood(consolidate_two, obs, obsl, bg_scores, cons_two_mix),[""])
    dmodel = ICA_PWM_Model("d", distance_model, 3, 3:4, trues(3,3), IPM_likelihood(distance_model, obs, obsl, bg_scores, trues(3,3)),[""])

    dpath=randstring()
    serialize(dpath, dmodel)
    drec=Model_Record(dpath,dmodel.log_Li)

    #check distance calculation
    pwmtest_1=zeros(1,4);pwmtest_1[1]=1.
    pwmtest_2=zeros(1,4);pwmtest_2[2]=1.
    @test pwm_distance(log.(pwmtest_1),log.(pwmtest_2)) == euclidean(pwmtest_1,pwmtest_2) == 1.4142135623730951

    #test consolidate check
    @test consolidate_check(distance_model) == (true,Dict{Integer,Vector{Integer}}())
    @test consolidate_check(consolidate_one) == (false, Dict{Integer,Vector{Integer}}(1=>[3]))
    @test consolidate_check(consolidate_two) == (false, Dict{Integer,Vector{Integer}}(1=>[2,3], 2=>[3]))

    #test overall consolidate function
    _,con_idxs=consolidate_check(consolidate_one)
    c1consmod=consolidate_srcs(con_idxs, c1model, obs, obsl, bg_scores, drec.log_Li, [drec])

    @test consolidate_check(c1consmod.sources)[1]
    @test c1consmod.log_Li > dmodel.log_Li
    @test all(c1consmod.mix_matrix[:,1])
    @test c1consmod.sources[1][1]==log.(src_ATG)
    @test c1consmod.sources[3][1]!=log.(src_ATG)
    @test c1consmod.flags==["consolidate from c1"]

    _,con_idxs=consolidate_check(consolidate_two)
    c2consmod=consolidate_srcs(con_idxs, c2model, obs, obsl, bg_scores, drec.log_Li, [drec])

    @test consolidate_check(c2consmod.sources)[1]
    @test c2consmod.log_Li > dmodel.log_Li
    @test all(c2consmod.mix_matrix[:,1])
    @test c2consmod.sources[1][1]==log.(src_ATG)
    @test c2consmod.sources[2][1]!=log.(src_ATG)
    @test c2consmod.sources[3][1]!=log.(src_ATG)
    @test c2consmod.flags==["consolidate from c2"]

    rm(dpath)
end

@testset "Model permutation functions" begin
    source_pwm = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    pwm_to_erode = [.25 .25 .25 .25
                    .97 .01 .01 .01
                    .01 .01 .01 .97
                    .01 .01 .97 .01
                    .25 .25 .25 .25]

    src_length_limits=2:5

    source_priors = assemble_source_priors(3, [source_pwm, source_pwm_2])
    mix_prior=0.2

    bg_scores = log.(fill(.25, (12,2)))
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATGATGATG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGATGATGA")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    test_model = ICA_PWM_Model("test", source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)

    ps_model= permute_source(test_model, Vector{Model_Record}(), obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000,weight_shift_freq=1.,length_change_freq=1.)
    @test ps_model.log_Li > test_model.log_Li
    @test ps_model.sources != test_model.sources
    @test ps_model.mix_matrix == test_model.mix_matrix
    @test "PS from test" in ps_model.flags

    pm_model= permute_mix(test_model, obs, obsl, bg_scores, test_model.log_Li, iterates=1000)
    @test pm_model.log_Li > test_model.log_Li
    @test pm_model.sources == test_model.sources
    @test pm_model.mix_matrix != test_model.mix_matrix
    @test "PM from test" in pm_model.flags

    psfm_model=perm_src_fit_mix(test_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors,  iterates=1000)
    @test psfm_model.log_Li > test_model.log_Li
    @test psfm_model.sources != test_model.sources
    @test psfm_model.mix_matrix != test_model.mix_matrix
    @test "PSFM from test" in psfm_model.flags

    fm_model=fit_mix(test_model,obs,obsl,bg_scores)
    @test fm_model.log_Li > test_model.log_Li
    @test fm_model.sources == test_model.sources
    @test fm_model.mix_matrix != test_model.mix_matrix
    @test "FM from test" in fm_model.flags
    @test "nofit" in fm_model.flags

    post_fm_psfm=perm_src_fit_mix(fm_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000)
    @test "PSFM from candidate" in post_fm_psfm.flags
    @test "nofit" in post_fm_psfm.flags

    rd_model=random_decorrelate(test_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000)
    @test rd_model.log_Li > test_model.log_Li
    @test rd_model.sources != test_model.sources
    @test rd_model.mix_matrix != test_model.mix_matrix
    @test "RD from test" in rd_model.flags

    rs_model=reinit_src(test_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000)
    @test rs_model.log_Li > test_model.log_Li
    @test rs_model.sources != test_model.sources
    @test rs_model.mix_matrix != test_model.mix_matrix
    @test "RS from test" in rs_model.flags

    erosion_sources=[(log.(source_pwm),1),(log.(source_pwm_2),1),(log.(pwm_to_erode),1)]

    eroded_mix=trues(2,3)

    erosion_lh=IPM_likelihood(erosion_sources,obs,obsl, bg_scores, eroded_mix)

    erosion_model=ICA_PWM_Model("erode", erosion_sources, test_model.informed_sources, test_model.source_length_limits,eroded_mix, erosion_lh, [""])

    eroded_model=erode_model(erosion_model, Vector{Model_Record}(), obs, obsl, bg_scores, erosion_model.log_Li)
    @test eroded_model.log_Li > erosion_model.log_Li
    @test eroded_model.sources != erosion_model.sources
    @test eroded_model.mix_matrix == erosion_model.mix_matrix
    @test "EM from erode" in eroded_model.flags
    @test eroded_model.sources[1]==erosion_model.sources[1]
    @test eroded_model.sources[2]==erosion_model.sources[2]
    @test eroded_model.sources[3]!=erosion_model.sources[3]
    @test eroded_model.sources[3][1]==erosion_model.sources[3][1][2:4,:]

    merger_srcs=   [([.1 .7 .1 .1
    .1 .7 .1 .1
    .15 .35 .35 .15],
    1
    ),
    ([.1 .7 .1 .1
        .1 .65 .6 .1
        .1 .7 .1 .1],
        1
    ),
    ([.1 .7 .1 .1
        .1 .7 .1 .1
        .1 .7 .1 .1
        .7 .1 .1 .1],
        1
    )]

    accurate_srcs=[ ([0.93 0.04 0.02 0.02; 0.02 0.02 0.02 0.94; 0.02 0.02 0.94 0.02], 1),
    ([0.94 0.02 0.02 0.02; 0.02 0.02 0.02 0.94; 0.02 0.02 0.94 0.02], 1),
    ([0.95 0.01 0.02 0.02; 0.02 0.02 0.02 0.94; 0.02 0.02 0.94 0.02], 1)]

    merger_mix = BitMatrix([true false false
    true true false])

    accurate_mix=BitMatrix([true false false
    true true false])

    merger_base=ICA_PWM_Model("merge", merger_srcs, 0,src_length_limits, merger_mix, IPM_likelihood(merger_srcs,obs,obsl, bg_scores, merger_mix),[""])

    merger_target=ICA_PWM_Model("target", accurate_srcs, 0, src_length_limits, accurate_mix, IPM_likelihood(accurate_srcs,obs,obsl,bg_scores,accurate_mix),[""])

    path=randstring()
    test_record = Model_Record(path, merger_target.log_Li)
    serialize(path, merger_target)

    dm_model=distance_merge(merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, iterates=1000)
    @test dm_model.log_Li > merger_base.log_Li
    @test dm_model.sources != merger_base.sources
    distance_dict=Dict(1=>3,2=>1,3=>1)
    for (n,src) in enumerate(dm_model.sources)
        @test (dm_model.sources[n]==merger_base.sources[n]) || (dm_model.sources[n]==merger_target.sources[distance_dict[n]])
    end
    @test "DM from merge" in dm_model.flags

    sm_model=similarity_merge(merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, iterates=1000)
    @test sm_model.log_Li > merger_base.log_Li
    @test sm_model.sources != merger_base.sources
    for (n,src) in enumerate(dm_model.sources)
        @test (sm_model.sources[n]==merger_base.sources[n]) || (sm_model.sources[n]==merger_target.sources[n])
    end
    @test "SM from merge" in sm_model.flags

    testwk=addprocs(1)[1]
    @everywhere import BioMotifInference

    ddm_model=remotecall_fetch(distance_merge, testwk, merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, iterates=1000, remote=true)
    @test ddm_model.log_Li > merger_base.log_Li
    @test ddm_model.sources != merger_base.sources
    for (n,src) in enumerate(ddm_model.sources)
        @test (ddm_model.sources[n]==merger_base.sources[n]) || (ddm_model.sources[n]==merger_target.sources[distance_dict[n]])
    end
    @test ("DM from merge" in ddm_model.flags)

    dsm_model=remotecall_fetch(similarity_merge, testwk, merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, iterates=1000, remote=true)
    @test dsm_model.log_Li > merger_base.log_Li
    @test dsm_model.sources != merger_base.sources
    for (n,src) in enumerate(dm_model.sources)
        @test (dsm_model.sources[n]==merger_base.sources[n]) || (dsm_model.sources[n]==merger_target.sources[n])
    end
    @test "SM from merge" in dsm_model.flags
    
    rmprocs(testwk)
    rm(path)
end

@testset "Ensemble assembly and nested sampling functions" begin
    ensembledir = randstring()
    spensembledir = randstring()
    distdir = randstring()

    source_pwm = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    source_pwm_2 = [.6 .1 .1 .2
    .2 .1 .1 .6
    .1 .2 .6 .1]

    src_length_limits=2:12
    no_sources=4

    source_priors = assemble_source_priors(no_sources, [source_pwm, source_pwm_2])
    mix_prior=.5

    bg_scores = log.(fill(.1, (30,27)))
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("CCGTTGACGATGTGATGAATAATGAAAGAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGACCGTTGACCAGATGGATG")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGACCCCGATTTTGAAAAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATCATGCTGATGATGAATCAGATGAAAG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGAATCTGACCCAGATGCCGATTTTGA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATTTTGATCAGGATGAATAAAGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATGGGCTGATGAACCGTTGACGATGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATCCTGCTGACCCCGATTTCAGTGAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGAATAAAGTCATCCTGCATGTGAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCGTTGACGATGTGATGAATGATAAAGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGACCGATGTTGACGATGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGAATGCCCCGATTTTGAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATCATGCTGATGATGAATAAAGAAAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGAATCTGACCCCGATCAGTTTGAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATTTTGATCAGATGGATGAATAAAG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATGGGCTGAACCGTTGACAGCGATGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATCCTGCTCAGGACCCCGATTTTATGGA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGAATCAGAAAGTCATCCTGCATGTGA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCGTTGACCAGGATGTGATGAATAAAGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGCAGACCGTTGACGATGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGACAGTGACCCAGCCGATTTTGA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATCATGCTGAATGTGATGAATAAAAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGATGAATCTGAATGCCCCGATTTTGA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATATGTTTGATGACAGTGAATAAAG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATGATGGGCTGAACCGTTGACGATGAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCACAGTCCTGCTGACCCCGATTATGTTGA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGATGAATAAAGTCATGATCCTGCTGA")
    ]
    
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))

    ensemble = IPM_Ensemble(ensembledir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    ensemble = IPM_Ensemble(ensembledir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    sp_ensemble = IPM_Ensemble(spensembledir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)

    @test length(ensemble.models) == 200
    for model in ensemble.models
        @test -1950 < model.log_Li < -1200
    end

    @test length(sp_ensemble.models) == 200
    for model in sp_ensemble.models
        @test -1950 < model.log_Li < -1200
    end

    assembler=addprocs(1)

    @everywhere using BioMotifInference

    dist_ensemble=IPM_Ensemble(assembler, distdir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    dist_ensemble=IPM_Ensemble(assembler, distdir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    @test length(dist_ensemble.models) == 200
    for model in ensemble.models
        @test -1950 < model.log_Li < -1200
    end

    rmprocs(assembler)
    rm(distdir, recursive=true)

    instruct = Permute_Instruct(full_perm_funcvec, ones(length(full_perm_funcvec))./length(full_perm_funcvec),600,900)

    @info "Testing convergence displays..."
    sp_logZ = converge_ensemble!(sp_ensemble, instruct, 50000000000., wk_disp=true, tuning_disp=true, ens_disp=true, conv_plot=true, src_disp=true, lh_disp=true, liwi_disp=true, max_iterates=3)

    sp_ensemble=reset_ensemble(sp_ensemble)


    @info "Testing threaded convergence..."
    sp_logZ = converge_ensemble!(sp_ensemble, instruct, 50000000000., wk_disp=false, tuning_disp=false, ens_disp=false, conv_plot=false, src_disp=false, lh_disp=false, liwi_disp=false)

    @test length(sp_ensemble.models) == 200
    @test length(sp_ensemble.log_Li) == length(sp_ensemble.log_Xi) == length(sp_ensemble.log_wi) == length(sp_ensemble.log_Liwi) == length(sp_ensemble.log_Zi) == length(sp_ensemble.Hi) == sp_ensemble.model_counter-200
    for i in 1:length(sp_ensemble.log_Li)-1
        @test sp_ensemble.log_Li[i] <= sp_ensemble.log_Li[i+1]
    end
    for i in 1:length(sp_ensemble.log_Zi)-1
        @test sp_ensemble.log_Zi[i] <= sp_ensemble.log_Zi[i+1]
    end
    @test sp_logZ > -1500.0

    @info "Testing multiprocess convergence..."
    @info "Spawning worker pool..."
    worker_pool=addprocs(2, topology=:master_worker)
    @everywhere using BioMotifInference

    ####CONVERGE############
    final_logZ = converge_ensemble!(ensemble, instruct, worker_pool, 50000000000., backup=(true,250), wk_disp=false, tuning_disp=false, ens_disp=false, conv_plot=false, src_disp=false, lh_disp=false, liwi_disp=false)

    rmprocs(worker_pool)

    @test length(ensemble.models) == 200
    @test length(ensemble.log_Li) == length(ensemble.log_Xi) == length(ensemble.log_wi) == length(ensemble.log_Liwi) == length(ensemble.log_Zi) == length(ensemble.Hi) == ensemble.model_counter-200
    for i in 1:length(ensemble.log_Li)-1
        @test ensemble.log_Li[i] <= ensemble.log_Li[i+1]
    end
    for i in 1:length(ensemble.log_Zi)-1
        @test ensemble.log_Zi[i] <= ensemble.log_Zi[i+1]
    end
    @test typeof(final_logZ) == Float64
    @test final_logZ > -1500.0

    @info "Tests complete!"

    rm(ensembledir, recursive=true)
    rm(spensembledir, recursive=true)
end