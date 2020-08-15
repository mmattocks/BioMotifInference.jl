module BioMotifInference
    using BioBackgroundModels, BioSequences, Distributed, Distributions, Serialization, UnicodePlots
    import DataFrames:DataFrame
    import ProgressMeter: AbstractProgress, Progress, @showprogress, next!, move_cursor_up_while_clearing_lines, printover, durationstring
    import Printf: @sprintf
    import StatsFuns: logaddexp, logsumexp #both are needed as logsumexp for two terms is deprecated
    import Random: rand, seed!, shuffle!
    import Distances: euclidean

    #CONSTANTS AND PERMUTE FUNCTION ARGUMENT DEFAULTS GIVING RISE TO IMPLEMENTATION-SPECIFIC SAMPLING EFFECTS
    global TUNING_MEMORY=50 #coefficient multiplied by function call limit to give total number of calls remembered by tuner
    global CONVERGENCE_MEMORY=500 #number of iterates to display for convergence interval history
    global PWM_SHIFT_DIST=Weibull(.5,.1) #distribution of weight matrix permutation magnitudes
    global PWM_SHIFT_FREQ=.2 #proportion of positions in source to permute weight matrix
    global PWM_LENGTHPERM_FREQ=.2 #proportion of sources to permute length
    global LENGTHPERM_RANGE=1:3
    global MIN_MIX_PERMFREQ=.0001 #minimum proportion of mix positions to permute in relevant mix perm funcs
    global PRIOR_WT=3. #estimate prior dirichlets from product of this constant and sample "mle" wm
    global PRIOR_LENGTH_MASS=.8
    global EROSION_INFO_THRESH=1.
    
    include("IPM/ICA_PWM_Model.jl")
    export Model_Record
    export ICA_PWM_Model
    include("IPM/IPM_likelihood.jl")
    include("IPM/IPM_prior_utilities.jl")
    assemble_source_priors
    include("ensemble/IPM_Ensemble.jl")
    export IPM_Ensemble, assemble_IPMs
    include("ensemble/ensemble_utilities.jl")
    include("permutation/permute_utilities.jl")
    include("permutation/orthogonality_helper.jl")
    include("permutation/permute_functions.jl")
    export full_perm_funcvec
    include("permutation/permute_control.jl")
    export Permute_Instruct
    include("permutation/Permute_Tuner.jl")
    include("utilities/model_display.jl")
    include("utilities/worker_diagnostics.jl")
    include("utilities/ns_progressmeter.jl")
    include("utilities/synthetic_genome.jl")
    export synthetic_sample
    include("nested_sampler/nested_step.jl")
    include("nested_sampler/converge_ensemble.jl")
    export converge_ensemble!

end # module