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
    mix_prior=.5

    bg_scores = log.(fill(.25, (12,2)))
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATGATGATG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TGATGATGATGA")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    test_model = ICA_PWM_Model("test", source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)

    ps_model= permute_source(test_model, Vector{Model_Record}(), obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000,weight_shift_freq=.2,length_change_freq=.2)
    @test ps_model.log_Li > test_model.log_Li
    @test ps_model.sources != test_model.sources
    @test ps_model.mix_matrix == test_model.mix_matrix
    @test ps_model.origin == "PS from test"

    pm_model= permute_mix(test_model, obs, obsl, bg_scores, test_model.log_Li, iterates=1000)
    @test pm_model.log_Li > test_model.log_Li
    @test pm_model.sources == test_model.sources
    @test pm_model.mix_matrix != test_model.mix_matrix
    @test "PM from test" == pm_model.origin

    psfm_model=perm_src_fit_mix(test_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors,  iterates=1000)
    @test psfm_model.log_Li > test_model.log_Li
    @test psfm_model.sources != test_model.sources
    @test psfm_model.mix_matrix != test_model.mix_matrix
    @test "PSFM from test" == psfm_model.origin

    fm_model=fit_mix(test_model, Vector{Model_Record}(), obs,obsl,bg_scores)
    @test fm_model.log_Li > test_model.log_Li
    @test fm_model.sources == test_model.sources
    @test fm_model.mix_matrix != test_model.mix_matrix
    @test "FM from test" == fm_model.origin
    @test fit_mix in fm_model.permute_blacklist

    post_fm_psfm=perm_src_fit_mix(fm_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000)
    @test "PSFM from candidate" == post_fm_psfm.origin
    @test fit_mix in post_fm_psfm.permute_blacklist

    rd_model=random_decorrelate(test_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000)
    @test rd_model.log_Li > test_model.log_Li
    @test rd_model.sources != test_model.sources
    @test rd_model.mix_matrix != test_model.mix_matrix
    @test "RD from test" == rd_model.origin

    rs_model=reinit_src(test_model, Vector{Model_Record}(),obs, obsl, bg_scores, test_model.log_Li, source_priors, iterates=1000)
    @test rs_model.log_Li > test_model.log_Li
    @test rs_model.sources != test_model.sources
    @test rs_model.mix_matrix != test_model.mix_matrix
    @test "RS from test" == rs_model.origin

    erosion_sources=[(log.(source_pwm),1),(log.(source_pwm_2),1),(log.(pwm_to_erode),1)]

    eroded_mix=trues(2,3)

    erosion_lh=IPM_likelihood(erosion_sources,obs,obsl, bg_scores, eroded_mix)

    erosion_model=ICA_PWM_Model("erode", "", erosion_sources, test_model.source_length_limits,eroded_mix, erosion_lh)

    eroded_model=erode_model(erosion_model, Vector{Model_Record}(), obs, obsl, bg_scores, erosion_model.log_Li)
    @test eroded_model.log_Li > erosion_model.log_Li
    @test eroded_model.sources != erosion_model.sources
    @test eroded_model.mix_matrix == erosion_model.mix_matrix
    @test "EM from erode" == eroded_model.origin
    @test eroded_model.sources[1]==erosion_model.sources[1]
    @test eroded_model.sources[2]==erosion_model.sources[2]
    @test eroded_model.sources[3]!=erosion_model.sources[3]
    @test eroded_model.sources[3][1]==erosion_model.sources[3][1][2:4,:]

    dbl_eroded=erode_model(eroded_model, Vector{Model_Record}(), obs, obsl, bg_scores, eroded_model.log_Li)
    @test dbl_eroded.permute_blacklist == [erode_model]


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

    merger_base=ICA_PWM_Model("merge", "", merger_srcs,src_length_limits, merger_mix, IPM_likelihood(merger_srcs,obs,obsl, bg_scores, merger_mix))

    merger_target=ICA_PWM_Model("target", "", accurate_srcs, src_length_limits, accurate_mix, IPM_likelihood(accurate_srcs,obs,obsl,bg_scores,accurate_mix))

    path=randstring()
    test_record = Model_Record(path, merger_target.log_Li)
    serialize(path, merger_target)

    dm_model=distance_merge(merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li)
    @test dm_model.log_Li > merger_base.log_Li
    @test dm_model.sources != merger_base.sources
    distance_dict=Dict(1=>3,2=>1,3=>1)
    @test all([src==merger_base.sources[n] || src==merger_target.sources[distance_dict[n]] for (n,src) in enumerate(dm_model.sources)])
    @test "DM from merge" == dm_model.origin

    sm_model=similarity_merge(merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li)
    @test sm_model.log_Li > merger_base.log_Li
    @test sm_model.sources != merger_base.sources
    @test all([src==merger_base.sources[n] || src==merger_target.sources[n] for (n,src) in enumerate(sm_model.sources)])
    @test "SM from merge" == sm_model.origin

    shuffled_model=shuffle_sources(merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li)
    @test sum(shuffled_model.sources.==merger_base.sources)==length(shuffled_model.sources)-1
    for (s,src) in enumerate(shuffled_model.sources)
        @test src==merger_base.sources[s] || src==merger_target.sources[s]
        if src == merger_base.sources[s]
            @test shuffled_model.mix_matrix[:,s]==merger_base.mix_matrix[:,s]
        else
            @test shuffled_model.mix_matrix[:,s]==merger_target.mix_matrix[:,s]
        end
    end
    @test "SS from merge" == shuffled_model.origin

    acc_base_mix=falses(2,1);acc_base_mix[1,1]=true
    acc_merge_mix=falses(2,1);acc_merge_mix[2,1]=true

    acc_base=ICA_PWM_Model("accbase", "", [merger_srcs[1]],src_length_limits, acc_base_mix, IPM_likelihood([merger_srcs[1]],obs,obsl, bg_scores, acc_base_mix))
    acc_merge=ICA_PWM_Model("accmerge", "", [accurate_srcs[1]],src_length_limits, acc_merge_mix, IPM_likelihood([accurate_srcs[1]],obs,obsl, bg_scores, acc_merge_mix))

    accpath=randstring()
    acc_record = Model_Record(accpath, acc_merge.log_Li)
    serialize(accpath, acc_merge)
 
    acc_model=accumulate_mix(acc_base, [acc_record], obs, obsl, bg_scores, acc_merge.log_Li)
    @test acc_model.mix_matrix==trues(2,1)
    @test acc_model.sources == acc_merge.sources
    @test "AM from accbase" == acc_model.origin

    testwk=addprocs(1)[1]
    @everywhere import BioMotifInference

    ddm_model=remotecall_fetch(distance_merge, testwk, merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, remote=true)
    @test ddm_model.log_Li > merger_base.log_Li
    @test ddm_model.sources != merger_base.sources
    @test all([src==merger_base.sources[n] || src==merger_target.sources[distance_dict[n]] for (n,src) in enumerate(ddm_model.sources)])
    @test "DM from merge" == ddm_model.origin

    dsm_model=remotecall_fetch(similarity_merge, testwk, merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, remote=true)
    @test dsm_model.log_Li > merger_base.log_Li
    @test dsm_model.sources != merger_base.sources
    @test all([src==merger_base.sources[n] || src==merger_target.sources[n] for (n,src) in enumerate(dsm_model.sources)])
    @test "SM from merge" == dsm_model.origin

    dshuffled_model=remotecall_fetch(shuffle_sources, testwk, merger_base, [test_record], obs, obsl, bg_scores, merger_base.log_Li, remote=true)
    @test sum(dshuffled_model.sources.==merger_base.sources)==length(dshuffled_model.sources)-1
    for (s,src) in enumerate(dshuffled_model.sources)
        @test src==merger_base.sources[s] || src==merger_target.sources[s]
        if src == merger_base.sources[s]
            @test dshuffled_model.mix_matrix[:,s]==merger_base.mix_matrix[:,s]
        else
            @test dshuffled_model.mix_matrix[:,s]==merger_target.mix_matrix[:,s]
        end
    end
    @test "SS from merge" == dshuffled_model.origin

    dacc_model=remotecall_fetch(accumulate_mix, testwk, acc_base, [acc_record], obs, obsl, bg_scores, acc_merge.log_Li)
    @test dacc_model.mix_matrix==trues(2,1)
    @test dacc_model.sources == acc_merge.sources
    @test "AM from accbase" == dacc_model.origin
    
    rmprocs(testwk)
    rm(path)
    rm(accpath)
end
