module BioMotifInference
    using BioSequences, Distributed, Distributions, Serialization, UnicodePlots
    import DataFrames:DataFrame
    import ProgressMeter: AbstractProgress, Progress, @showprogress
    import Printf: @sprintf
    import StatsFuns: logaddexp, logsumexp #both are needed as logsumexp for two terms is deprecated
    import Random: rand, seed!, shuffle!
    import Distances: euclidean
    import BioBackgroundModels: lps

    include("IPM/ICA_PWM_Model.jl")
    include("IPM/IPM_likelihood.jl")
    include("IPM/IPM_prior_utilities.jl")
    include("ensemble/IPM_Ensemble.jl")
    include("ensemble/ensemble_utilities.jl")
    include("permutation/permute_utilities.jl")
    include("permutation/orthogonality_helper.jl")
    include("permutation/permute_functions.jl")
    include("permutation/permute_control.jl")
    include("permutation/tuning.jl")
    include("utilities/model_display.jl")
    include("utilities/ns_progressmeter.jl")
    include("nested_sampler.jl")
end # module