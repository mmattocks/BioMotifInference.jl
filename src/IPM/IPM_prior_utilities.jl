function read_fa_wms_tr(path::String)
    wms=Vector{Matrix{Float64}}()
    wm=zeros(1,4)
    f=open(path)
    for line in eachline(f)
        prefix=line[1:2]
        prefix == "01" && (wm=transpose([parse(AbstractFloat,i) for i in split(line)[2:end]]))
        prefix != "01" && prefix != "NA" && prefix != "PO" && prefix != "//" && (wm=vcat(wm, transpose([parse(AbstractFloat,i) for i in split(line)[2:end]])))
        prefix == "//" && push!(wms, wm)
    end
    return wms
end

#wm_samples are in decimal probability space, not log space
function assemble_source_priors(no_sources::Integer, wm_samples::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, prior_wt::AbstractFloat=PRIOR_WT) #estimate a dirichlet prior on wm_samples inputs; if the number of samples is lower than the number of sources, return a false bool for init and permutation functions
    source_priors = Vector{Union{Vector{Dirichlet{Float64}},Bool}}()
    for source in 1:no_sources
        if source <= length(wm_samples)
            push!(source_priors, estimate_dirichlet_prior_on_wm(wm_samples[source], prior_wt))
        else
            push!(source_priors, false)
        end
    end
    return source_priors
end

            function estimate_dirichlet_prior_on_wm(wm::AbstractMatrix{<:AbstractFloat}, wt::AbstractFloat=PRIOR_WT)
                for i in 1:size(wm)[1]
                    !(isprobvec(wm[i,:])) && throw(DomainError("Bad weight vec supplied to estimate_dirichlet_prior_on_wm! $(wm[i,:])"))
                end
                prior = Vector{Dirichlet{Float64}}()
                for position in 1:size(wm)[1]
                    normvec=wm[position,:]
                    zero_idxs=findall(isequal(0.),wm[position,:])
                    normvec[zero_idxs].+=10^-99
                    push!(prior, Dirichlet(normvec.*wt))
                end
                return prior
            end

function cluster_mix_prior!(df::DataFrame, wms::AbstractVector{<:AbstractMatrix{<:AbstractFloat}})
    mix=falses(size(df,1),length(wms))
    for (o, row) in enumerate(eachrow(df))
        row.cluster != 0 && (mix[o,row.cluster]=true)
    end
    
    represented_sources=unique(df.cluster)
    wms=wms[represented_sources]
    return mix[:,represented_sources]
end



function infocenter_wms_trim(wm::AbstractMatrix{<:AbstractFloat}, trimsize::Integer)
    !(size(wm,2)==4) && throw(DomainError("Bad wm! 2nd dimension should be size 4"))
    infovec=get_pwm_info(wm, logsw=false)
    maxval, maxidx=findmax(infovec)
    upstream_extension=Int(floor((trimsize-1)/2))
    downstream_extension=Int(ceil((trimsize-1)/2))
    1+upstream_extension+downstream_extension > size(wm,1) && throw(DomainError("Src too short for trim! $upstream_extension $downstream_extension"))
    return wm[max(1,maxidx-upstream_extension):min(maxidx+downstream_extension,size(wm,1)),:]
end

function filter_priors(target_src_no::Integer, target_src_size::Integer, prior_wms::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, prior_mix::BitMatrix)
    wms=Vector{Matrix{Float64}}(undef, target_src_no)
    freqsort_idxs=sortperm([sum(prior_mix[:,s]) for s in 1:length(prior_wms)])
    for i in 1:target_src_no
        target_src_idx=freqsort_idxs[i]
        wms[i]=infocenter_wms_trim(prior_wms[target_src_idx], target_src_size)
    end
    return wms
end

function combine_filter_priors(target_src_no::Integer, target_src_size::Integer, prior_wms::Tuple{<:AbstractVector{<:AbstractMatrix{<:AbstractFloat}},AbstractVector{<:AbstractMatrix{<:AbstractFloat}}}, prior_mix::Tuple{BitMatrix,BitMatrix})
    wms=Vector{Matrix{Float64}}(undef, target_src_no)
    cat_wms=vcat(prior_wms[1],prior_wms[2])
    first_freq=[sum(prior_mix[1][:,s]) for s in 1:length(prior_wms[1])]
    second_freq=[sum(prior_mix[2][:,s]) for s in 1:length(prior_wms[2])]
    freqsort_idxs=sortperm(vcat(first_freq,second_freq))
    for i in 1:target_src_no
        target_src_idx=freqsort_idxs[i]
        wms[i]=infocenter_wms_trim(cat_wms[target_src_idx], target_src_size)
    end
    return wms
end
