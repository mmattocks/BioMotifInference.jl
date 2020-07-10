module BioMotifInference
    using BioSequences, Distributed, Distributions, Serialization, UnicodePlots
    import DataFrames:DataFrame
    import ProgressMeter: AbstractProgress, Progress, @showprogress, next!, move_cursor_up_while_clearing_lines, printover, durationstring
    import Printf: @sprintf
    import StatsFuns: logaddexp, logsumexp #both are needed as logsumexp for two terms is deprecated
    import Random: rand, seed!, shuffle!
    import Distances: euclidean
    import BioBackgroundModels: lps

    include("IPM/ICA_PWM_Model.jl")
    export Model_Record
    export ICA_PWM_Model
    include("IPM/IPM_likelihood.jl")
    include("IPM/IPM_prior_utilities.jl")
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
    include("nested sampler/nested_step.jl")
    include("nested sampler/converge_ensemble.jl")

end # module