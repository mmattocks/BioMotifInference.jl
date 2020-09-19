
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
    no_sources=3

    source_priors = assemble_source_priors(no_sources, [source_pwm, source_pwm_2])
    mix_prior=.5

    bg_scores = log.(fill(.1, (30,4)))
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("CCGTTGACGATGTGATGAATAATGAAAGAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGACCGTTGACCAGATGGATG")
    BioSequences.LongSequence{DNAAlphabet{2}}("CCCCGATGATGACCCCGATTTTGAAAAAAA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCATCATGCTGATGATGAATCAGATGAAAG")
    ]

    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))

    ensemble = IPM_Ensemble(ensembledir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    ensemble = IPM_Ensemble(ensembledir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    sp_ensemble = IPM_Ensemble(spensembledir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits, posterior_switch=true)

    @test length(ensemble.models) == 200
    for model in ensemble.models
        @test -350 < model.log_Li < -150
    end

    @test length(sp_ensemble.models) == 200
    for model in sp_ensemble.models
        @test -350 < model.log_Li < -150
    end

    assembler=addprocs(1)

    @everywhere using BioMotifInference

    dist_ensemble=IPM_Ensemble(assembler, distdir, 150, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits)
    dist_ensemble=IPM_Ensemble(assembler, distdir, 200, source_priors, (falses(0,0),mix_prior), bg_scores, obs, src_length_limits) #test resumption

    @test length(dist_ensemble.models) == 200
    for model in ensemble.models
        @test -350 < model.log_Li < -150
    end

    rmprocs(assembler)
    rm(distdir, recursive=true)

    models_to_permute = 600
    funclimit=200
    funcvec=full_perm_funcvec

    instruct = Permute_Instruct(funcvec, ones(length(funcvec))./length(funcvec),models_to_permute,200, min_clmps=fill(.02,length(funcvec)))

    @info "Testing convergence displays..."
    sp_logZ = converge_ensemble!(sp_ensemble, instruct, converge_factor=500.,  wk_disp=true, tuning_disp=true, ens_disp=true, conv_plot=true, src_disp=true, lh_disp=true, liwi_disp=true, max_iterates=50)

    sp_ensemble=reset_ensemble!(sp_ensemble)

    @info "Testing threaded convergence..."
    sp_logZ = converge_ensemble!(sp_ensemble, instruct,  converge_factor=500.,wk_disp=false, tuning_disp=false, ens_disp=false, conv_plot=false, src_disp=false, lh_disp=false, liwi_disp=false, backup=(true, 150), max_iterates=300)

    @info "Testing resumption..."
    sp_logZ = converge_ensemble!(sp_ensemble, instruct,  converge_factor=500.,wk_disp=false, tuning_disp=false, ens_disp=false, conv_plot=false, src_disp=false, lh_disp=false, liwi_disp=false, backup=(true, 150))
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
    final_logZ = converge_ensemble!(ensemble, instruct, worker_pool,  converge_factor=500., wk_disp=false, tuning_disp=false, ens_disp=false, conv_plot=false, src_disp=false, lh_disp=false, liwi_disp=false, backup=(true,500), clean=(true, 500, 1000))

    convits=length(ensemble.log_Li)

    #test converging already converged, wih different converge criterion
    final_logZ = converge_ensemble!(ensemble, instruct, worker_pool,  converge_factor=150., converge_criterion="compression", wk_disp=false, tuning_disp=false, ens_disp=false, conv_plot=false, src_disp=false, lh_disp=false, liwi_disp=false, backup=(true,500), clean=(true, 500, 1000))

    length(ensemble.log_Li)==convits

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

    rm(ensembledir, recursive=true)
    rm(spensembledir, recursive=true)
end