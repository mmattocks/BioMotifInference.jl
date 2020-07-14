struct Model_Record #record struct to associate a log_Li with a saved, calculated model
    path::String
    log_Li::AbstractFloat
end

struct ICA_PWM_Model #Independent component analysis position weight matrix model
    name::String #designator for saving model to posterior
    sources::Vector{Tuple{<:AbstractMatrix{<:AbstractFloat},<:Integer}} #vector of PWM signal sources (LOG PROBABILITY!!!) tupled with an index denoting the position of the first PWM base on the prior matrix- allows us to permute length and redraw from the appropriate prior position
    informed_sources::Integer #number of sources with informative priors- these are not subject to frequency sorting in model mergers
    source_length_limits::UnitRange{<:Integer} #min/max source lengths for init and permutation
    mix_matrix::BitMatrix # obs x sources bool matrix
    log_Li::AbstractFloat
    flags::Vector{String} #optional flags for search patterns
end

#ICA_PWM_Model FUNCTIONS
ICA_PWM_Model(name::String, source_priors::AbstractVector{<:AbstractVector{Dirichlet{Float64}}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractArray{<:AbstractFloat}, observations::AbstractArray{<:Integer}, source_length_limits::UnitRange{<:Integer}) = init_IPM(name, source_priors,mix_prior,bg_scores,observations,source_length_limits)

#MODEL INIT
function init_IPM(name::String, source_priors::AbstractVector{<:AbstractVector{<:Dirichlet{<:AbstractFloat}}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractArray{<:AbstractFloat}, observations::AbstractArray{<:Integer}, source_length_limits::UnitRange{<:Integer})
    assert_obs_bg_compatibility(observations,bg_scores)
    T,O = size(observations)
    S=length(source_priors)
    obs_lengths=[findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]
    sources=init_logPWM_sources(source_priors, source_length_limits)
    mix=init_mix_matrix(mix_prior,O,S)
    log_lh = IPM_likelihood(sources, observations, obs_lengths, bg_scores, mix)

   return ICA_PWM_Model(name, sources, size(mix_prior[1],2), source_length_limits, mix, log_lh, ["init"])
end
                #init_IPM SUBFUNCS
                function assert_obs_bg_compatibility(obs, bg_scores)
                    T,O=size(obs)
                    t,o=size(bg_scores)
                    O!=o && throw(DomainError("Background scores and observations must have same number of observation columns!"))
                    T!=t+1 && throw(DomainError("Background score array must have the same observation lengths as observations!"))
                end

                function init_logPWM_sources(prior_vector::AbstractVector{<:AbstractVector{<:Dirichlet{<:AbstractFloat}}}, source_length_limits::UnitRange{<:Integer})
                    srcvec = Vector{Tuple{Matrix{Float64},Int64}}()
                    prior_coord = 1
                        for (p, prior) in enumerate(prior_vector)
                            min_PWM_length=source_length_limits[1]
                            PWM_length = rand(min_PWM_length:length(prior)) #generate a PWM from a random subset of the prior
                            PWM = zeros(PWM_length,4)
                            prior_coord = rand(1:length(prior)-PWM_length+1) #determine what position on the prior to begin sampling from based on the PWM length
                            for (position, dirichlet) in enumerate(prior)
                                if position >= prior_coord #skip prior positions that are before the selected prior_coord
                                    sample_coord = min(position-prior_coord+1,PWM_length) #sample_coord is the position on the sampled PWM
                                    PWM[sample_coord, :] = rand(dirichlet) #draw the position WM from the dirichlet
                                    !isprobvec(PWM[sample_coord, :]) && throw(DomainError("Bad weight vec produced by init_sources! $(PWM[sample_coord,:])"))#make sure it's a valid probvec
                                end
                            end
                            push!(srcvec, (log.(PWM), prior_coord)) #push the source PWM to the source vector with the prior coord idx to allow drawing from the appropriate prior dirichlets on permuting source length
                        end
                    return srcvec
                end

                function init_mix_matrix(mix_prior::Tuple{BitMatrix,<:AbstractFloat}, no_observations::Integer, no_sources::Integer)
                    inform,uninform=mix_prior
                    if size(inform,2) > 0
                        @assert size(inform,1)==no_observations && size(inform,2)<=no_sources "Bad informative mix prior dimensions!"
                    end
                    @assert 0.0 <= uninform <=1.0 "Uninformative mix prior not between 0.0 and 1.0!"
                    mix_matrix = falses(no_observations, no_sources)
                    if size(inform,2)>0
                        mix_matrix[:,1:size(inform,2)]=inform
                    end
                    for index in CartesianIndices((1:no_observations,size(inform,2)+1:no_sources))
                        rand() <= uninform && (mix_matrix[index] = true)
                    end
                    return mix_matrix
                end

function Base.show(io::IO, m::ICA_PWM_Model; nsrc::Integer=length(m.sources), progress=false)
    nsrc == 0 && (nsrc=length(m.sources))
    nsrc>length(m.sources) && (nsrc=length(m.sources))
    nsrc==length(m.sources) ? (srcstr="All") : (srcstr="Top $nsrc")

    printidxs,printsrcs,printfreqs=sort_sources(m,nsrc)

    printstyled(io, "ICA PWM Model $(m.name) w/ logLi $(m.log_Li)\n", bold=true)
    println(io, srcstr*" sources:")
    for src in 1:nsrc
        print(io, "S$(printidxs[src]), $(printfreqs[src]*100)%: ")
        pwmstr_to_io(io, printsrcs[src])
        println(io)
    end

    progress && return(nsrc+3)
end

function sort_sources(m, nsrc)
    printidxs=Vector{Integer}()
    printsrcs=Vector{Matrix{Float64}}()
    printfreqs=Vector{Float64}()

    freqs=vec(sum(m.mix_matrix,dims=1)); total=size(m.mix_matrix,1)
    sortfreqs=sort(freqs,rev=true); sortidxs=sortperm(freqs,rev=true)
    for srcidx in 1:nsrc
        push!(printidxs, sortidxs[srcidx])
        push!(printsrcs, m.sources[sortidxs[srcidx]][1])
        push!(printfreqs, sortfreqs[srcidx]/total)
    end
    return printidxs,printsrcs,printfreqs
end