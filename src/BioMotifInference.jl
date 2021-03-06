module BioMotifInference
    using BioBackgroundModels, BioSequences, Distributed, Distributions, Serialization, UnicodePlots
    import DataFrames:DataFrame
    import ProgressMeter: AbstractProgress, Progress, @showprogress, next!, move_cursor_up_while_clearing_lines, printover, durationstring
    import Printf: @sprintf
    import StatsFuns: logaddexp, logsumexp, logsubexp
    import Random: rand, seed!, shuffle!
    import Distances: euclidean
    import Measurements: measurement

    #CONSTANTS AND PERMUTE FUNCTION ARGUMENT DEFAULTS GIVING RISE TO IMPLEMENTATION-SPECIFIC SAMPLING EFFECTS
    global MOTIF_EXPECT=1. #motif expectation- this value to be divided by obs lengths to obtain penalty factor for scoring

    global REVCOMP=true #calculate scores on both strands? 

    global TUNING_MEMORY=20 #coefficient multiplied by Permute_Instruct function call limit to give total number of calls remembered by tuner per function
    global CONVERGENCE_MEMORY=500 #number of iterates to display for convergence interval history

    global SRC_PERM_FREQ=.5 #frequency with which random_decorrelate will permute a source

    global PWM_SHIFT_DIST=Weibull(.5,.1) #distribution of weight matrix permutation magnitudes
    global PWM_SHIFT_FREQ=.2 #proportion of positions in source to permute weight matrix
    global PWM_LENGTHPERM_FREQ=.2 #proportion of sources to permute length
    global LENGTHPERM_RANGE=1:3

    global PRIOR_WT=3. #estimate prior dirichlets from product of this constant and sample "mle" wm
    global PRIOR_LENGTH_MASS=.8

    global EROSION_INFO_THRESH=1.

    global CONSOLIDATE_THRESH=.035
    
    include("IPM/ICA_PWM_Model.jl")
    export Model_Record
    export ICA_PWM_Model
    include("IPM/IPM_likelihood.jl")
    include("IPM/IPM_prior_utilities.jl")
    assemble_source_priors
    include("ensemble/IPM_Ensemble.jl")
    export IPM_Ensemble, assemble_IPMs
    include("permutation/permute_utilities.jl")
    include("permutation/orthogonality_helper.jl")
    include("permutation/permute_functions.jl")
    export full_perm_funcvec
    include("permutation/permute_control.jl")
    export Permute_Instruct
    include("permutation/Permute_Tuner.jl")
    include("ensemble/ensemble_utilities.jl")
    export ensemble_history, reset_ensemble!, move_ensemble!, copy_ensemble!, rewind_ensemble, complete_evidence, get_model, show_models, reestimate_ensemble!
    include("utilities/model_display.jl")
    include("utilities/worker_diagnostics.jl")
    include("utilities/ns_progressmeter.jl")
    include("utilities/synthetic_genome.jl")
    include("utilities/worker_sequencer.jl")
    export synthetic_sample
    include("nested_sampler/nested_step.jl")
    include("nested_sampler/converge_ensemble.jl")
    export converge_ensemble!

end # module