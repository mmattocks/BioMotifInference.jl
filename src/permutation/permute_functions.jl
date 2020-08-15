#DECORRELATION SEARCH PATTERNS
#random permutation of single sources until model with log likelihood>contour found or iterates limit reached. will always produce at least one weight shift per iteration for weight_shift_freq>0, more than this or length changes depend on the supplied probabilities. one length change per iterate
function permute_source(m::ICA_PWM_Model, models::Vector{Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},Bool}}; iterates::Integer=length(m.sources)*2, weight_shift_freq::AbstractFloat=PWM_SHIFT_FREQ, length_change_freq::AbstractFloat=PWM_LENGTHPERM_FREQ, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=PWM_SHIFT_DIST, remote=false) 
#weight_shift_dist is given in decimal probability values- converted to log space in permute_source_lengths!
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources);

    a, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources);
        s = rand(1:S)
        clean=Vector{Bool}(trues(O))
        clean[m.mix_matrix[:,s]].=false #all obs with source are dirty

        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))
        new_log_Li, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate", "PS from $(m.name)", new_sources, m.source_length_limits, m.mix_matrix, new_log_Li)) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate", "PS from $(m.name)", new_sources, m.source_length_limits, m.mix_matrix, new_log_Li), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

function permute_mix(m::ICA_PWM_Model, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat; iterates::Integer=10, mix_move_range::UnitRange=Int(ceil(MIN_MIX_PERMFREQ*length(m.mix_matrix))):length(m.mix_matrix), remote=false) 
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_mix=falses(size(m.mix_matrix))

    a, cache = IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true)
    dirty=false

    while (new_log_Li <= contour || !dirty) && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        mix_moves=rand(mix_move_range)
        mix_moves > length(m.mix_matrix) && (mix_moves = length(m.mix_matrix))
    
        new_mix, clean = mix_matrix_decorrelate(m.mix_matrix, mix_moves) #generate a decorrelated candidate mix
        c_log_li, c_cache = IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #calculate the model with the candidate mix
        positive_indices=c_cache.>(cache) #obtain any obs indices that have greater probability than we started with
        clean=Vector{Bool}(trues(O)-positive_indices)
        if any(positive_indices) #if there are any such indices
            new_log_Li, cache = IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #calculate the new model
            dirty=true
        end
        iterate += 1
    end

    return ICA_PWM_Model("candidate","PM from $(m.name)", m.sources, m.source_length_limits, new_mix, new_log_Li) #no consolidate check is necessary as sources havent changed
end

function perm_src_fit_mix(m::ICA_PWM_Model,  models::Vector{Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat},contour::AbstractFloat,  source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},Bool}}; iterates::Integer=length(m.sources)*2, weight_shift_freq::AbstractFloat=PWM_SHIFT_FREQ, length_change_freq::AbstractFloat=PWM_LENGTHPERM_FREQ, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=PWM_SHIFT_DIST, remote=false)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    fit_mix in m.permute_blacklist ? (new_bl=m.permute_blacklist) : (new_bl=Vector{Function}())

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix); tm_one=deepcopy(m.mix_matrix);
        clean=Vector{Bool}(trues(O))
        s = rand(1:S); 
        clean[new_mix[:,s]].=false #all obs starting with source are dirty

        new_mix[:,s].=false;tm_one[:,s].=true

        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))

        l,zero_cache=IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true)
        l,one_cache=IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, tm_one, true, true)

        fit_mix=one_cache.>=zero_cache

        new_mix[:,s]=fit_mix

        clean[fit_mix].=false #all obs ending with source are dirty

        new_log_Li = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, false, zero_cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate","PSFM from $(m.name)",new_sources, m.source_length_limits, new_mix, new_log_Li, new_bl)) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate","PSFM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li, new_bl), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

function fit_mix(m::ICA_PWM_Model, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}; exclude_src::Integer=0, remote=false)
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_mix=deepcopy(m.mix_matrix); test_mix=falses(size(m.mix_matrix))
    new_bl=[fit_mix]

    l,zero_cache=IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, test_mix, true, true)
    for (s,source) in enumerate(m.sources)
        if s!=exclude_src
            test_mix=falses(size(m.mix_matrix))
            test_mix[:,s].=true
            l,src_cache=IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, test_mix, true, true)
            fit_mix=src_cache.>=zero_cache
            new_mix[:,s]=fit_mix
        end
    end
    new_mix==m.mix_matrix ? new_log_Li=-Inf : new_log_Li = IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, new_mix, true)

    return ICA_PWM_Model("candidate","FM from $(m.name)",m.sources, m.source_length_limits, new_mix, new_log_Li, new_bl) #no consolidate check necessary as no change to sources
end

function random_decorrelate(m::ICA_PWM_Model, models::Vector{Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},Bool}}; iterates::Integer=length(m.sources)*2, weight_shift_freq::AbstractFloat=PWM_SHIFT_FREQ, length_change_freq::AbstractFloat=PWM_LENGTHPERM_FREQ, weight_shift_dist::Distributions.ContinuousUnivariateDistribution=PWM_SHIFT_DIST, mix_move_range::UnitRange=1:Int(ceil(size(m.mix_matrix,1)*MIN_MIX_PERMFREQ)), remote=false)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    a, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))
        s = rand(1:length(m.sources))
        clean[new_mix[:,s]].=false #all obs starting with source are dirty
        weight_shift_freq > 0 && (new_sources[s]=permute_source_weights(new_sources[s], weight_shift_freq, weight_shift_dist))
        rand() < length_change_freq && (new_sources[s]=permute_source_length(new_sources[s], source_priors[s], m.source_length_limits))
        new_mix[:,s]=mixvec_decorrelate(new_mix[:,s],rand(mix_move_range))
        clean[new_mix[:,s]].=false #all obs ending with source are dirty
        new_log_Li, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true, cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate", "RD from $(m.name)", new_sources, m.source_length_limits, new_mix, new_log_Li)) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate","RD from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

function distance_merge(m::ICA_PWM_Model, models::AbstractVector{<:Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat; iterates::Integer=length(m.sources)*2, remote=false)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    a, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))

        remote ? (merger_m = deserialize(rand(models).path)) : (merger_m = remotecall_fetch(deserialize, 1, rand(models).path)) #randomly select a model to merge
        s = rand(1:S) #randomly select a source to merge
        merge_s=most_dissimilar(new_mix[:,s],merger_m.mix_matrix)
        
        clean[new_mix[:,s]].=false #mark dirty any obs that start with the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source
        new_mix[:,s] = merger_m.mix_matrix[:,merge_s] #copy the mixvector (without whichj the source will likely be highly improbable)
        clean[new_mix[:,s]].=false #mark dirty any obs that end with the source

        new_log_Li, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate","DM from $(m.name)",new_sources, m.source_length_limits, new_mix, new_log_Li,Vector{Function}())) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate","DM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

function similarity_merge(m::ICA_PWM_Model, models::AbstractVector{<:Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat; iterates::Integer=length(m.sources)*2, remote=false)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources)

    a, cache = IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))

        remote ? (merger_m = deserialize(rand(models).path)) : (merger_m = remotecall_fetch(deserialize, 1, rand(models).path))#randomly select a model to merge
        s = rand(1:S) #randomly select a source in the model to merge
        merge_s=most_similar(m.mix_matrix[:,s],merger_m.mix_matrix)

        clean[m.mix_matrix[:,s]].=false #mark dirty any obs that have the source
        new_sources[s] = merger_m.sources[merge_s] #copy the source, but dont copy the mixvector
        new_log_Li, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate","SM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li,Vector{Function}())) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate","SM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

function reinit_src(m::ICA_PWM_Model, models::AbstractVector{<:Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat},contour::AbstractFloat, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},Bool}}; iterates::Integer=length(m.sources)*2, remote=false)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix); tm_one=deepcopy(m.mix_matrix);
        clean=Vector{Bool}(trues(O))
        s = rand(1:S); 
        clean[new_mix[:,s]].=false #all obs starting with source are dirty

        new_mix[:,s].=false;tm_one[:,s].=true

        new_sources[s] = init_logPWM_sources([source_priors[s]], m.source_length_limits)[1] #reinitialize the source

        l,zero_cache=IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true)
        l,one_cache=IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, tm_one, true, true)

        fit_mix=one_cache.>=zero_cache

        new_mix[:,s]=fit_mix

        clean[fit_mix].=false #all obs ending with source are dirty

        new_log_Li = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, false, zero_cache, clean)
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate","RS from $(m.name)", new_sources, m.source_length_limits, new_mix, new_log_Li,Vector{Function}())) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate","RS from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

function erode_model(m::ICA_PWM_Model, models::AbstractVector{<:Model_Record}, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat; info_thresh::AbstractFloat=EROSION_INFO_THRESH, remote=false)
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources)

    erosion_sources=Set{Integer}()
    for (s,src) in enumerate(m.sources)
        pwm,pi=src
        if size(pwm,1)>m.source_length_limits[1]
            infovec=get_pwm_info(pwm)
            any(info-><(info, info_thresh),infovec) && push!(erosion_sources,s)
        end
    end

    length(erosion_sources)==0 && return ICA_PWM_Model("candidate","EM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li,Vector{Function}())#if we got a model we cant erode bail out with a model marked -Inf lh

    a, cache = IPM_likelihood(m.sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true)

    while new_log_Li <= contour && length(erosion_sources) > 0 #until we produce a model more likely than the lh contour or there are no more sources to erode
        clean=Vector{Bool}(trues(O))
        s=pop!(erosion_sources)

        new_sources[s]=erode_source(new_sources[s], m.source_length_limits, info_thresh)
        clean[m.mix_matrix[:,s]].=false

        new_log_Li, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, m.mix_matrix, true, true, cache, clean) #assess likelihood
        iterate += 1
    end

    cons_check, cons_idxs = consolidate_check(new_sources)
    cons_check ? (return ICA_PWM_Model("candidate","EM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li,Vector{Function}())) : (return consolidate_srcs(cons_idxs, ICA_PWM_Model("candidate","EM from $(m.name)",new_sources, m.source_length_limits, m.mix_matrix, new_log_Li), obs_array, obs_lengths, bg_scores, contour, models; remote=remote))
end

full_perm_funcvec=[permute_source, permute_mix, perm_src_fit_mix, fit_mix, random_decorrelate, distance_merge, similarity_merge, reinit_src, erode_model]