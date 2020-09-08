@info "Loading test packages..."

using BioMotifInference, BioBackgroundModels, BioSequences, Distributions, Distributed, Random, Serialization, Test
import StatsFuns: logsumexp, logaddexp
import BioMotifInference:estimate_dirichlet_prior_on_wm, assemble_source_priors, init_logPWM_sources, wm_shift, permute_source_weights, get_length_params, permute_source_length, get_pwm_info, get_erosion_idxs, erode_source, init_mix_matrix, mixvec_decorrelate, mix_matrix_decorrelate, most_dissimilar, most_similar, revcomp_pwm, score_sources_ds!, score_sources_ss!, weave_scores_ss!, weave_scores_ds!, IPM_likelihood, consolidate_check, consolidate_srcs, pwm_distance, permute_source, permute_mix, perm_src_fit_mix, fit_mix, random_decorrelate, reinit_src, erode_model, reinit_src, shuffle_sources, accumulate_mix, distance_merge, similarity_merge, converge_ensemble!, reset_ensemble!, Permute_Tuner, PRIOR_WT, TUNING_MEMORY, CONVERGENCE_MEMORY, tune_weights!, update_weights!, clamp_pvec!
import Distances: euclidean

@info "Beginning tests..."
using Random
Random.seed!(786)
O=1000;S=50

include("pwm_unit_tests.jl")
include("mix_matrix_unit_tests.jl")
include("likelihood_unit_tests.jl")
include("consolidate_unit_tests.jl")
include("permute_func_tests.jl")
include("permute_tuner_tests.jl")
include("ensemble_tests.jl")
@info "Tests complete!"
