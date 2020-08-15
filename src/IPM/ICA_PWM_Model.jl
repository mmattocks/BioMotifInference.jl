struct Model_Record #record struct to associate a log_Li with a saved, calculated model
    path::String
    log_Li::AbstractFloat
end

struct ICA_PWM_Model #Independent component analysis position weight matrix model
    name::String #designator for saving model to posterior
    origin::String #functions instantiating IPMs should give an informative desc of the means by which the model was generated
    sources::Vector{Tuple{<:AbstractMatrix{<:AbstractFloat},<:Integer}} #vector of PWM signal sources (LOG PROBABILITY!!!) tupled with an index denoting the position of the first PWM base on the prior matrix- allows us to permute length and redraw from the appropriate prior position
    source_length_limits::UnitRange{<:Integer} #min/max source lengths for init and permutation
    mix_matrix::BitMatrix # obs x sources bool matrix
    log_Li::AbstractFloat
    permute_blacklist::Vector{Function} #blacklist of functions that ought not be used to permute this model (eg. because to do so would not generate a different model for IPMs produced from fitting the mix matrix)
    function ICA_PWM_Model(name, origin, sources, source_length_limits, mix_matrix, log_Li, permute_blacklist=Vector{Function}())
        new(name, origin, sources, source_length_limits, mix_matrix, log_Li, permute_blacklist)
    end
end

#ICA_PWM_Model FUNCTIONS
ICA_PWM_Model(name::String, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractArray{<:AbstractFloat}, observations::AbstractArray{<:Integer}, source_length_limits::UnitRange{<:Integer}) = init_IPM(name, source_priors,mix_prior,bg_scores,observations,source_length_limits)

#MODEL INIT
function init_IPM(name::String, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractArray{<:AbstractFloat}, observations::AbstractArray{<:Integer}, source_length_limits::UnitRange{<:Integer})
    assert_obs_bg_compatibility(observations,bg_scores)
    T,O = size(observations)
    S=length(source_priors)
    obs_lengths=[findfirst(iszero,observations[:,o])-1 for o in 1:size(observations)[2]]
    sources=init_logPWM_sources(source_priors, source_length_limits)
    mix=init_mix_matrix(mix_prior,O,S)
    log_lh = IPM_likelihood(sources, observations, obs_lengths, bg_scores, mix)

   return ICA_PWM_Model(name, "init_IPM", sources, source_length_limits, mix, log_lh)
end
                #init_IPM SUBFUNCS
                function assert_obs_bg_compatibility(obs, bg_scores)
                    T,O=size(obs)
                    t,o=size(bg_scores)
                    O!=o && throw(DomainError("Background scores and observations must have same number of observation columns!"))
                    T!=t+1 && throw(DomainError("Background score array must have the same observation lengths as observations!"))
                end

                function init_logPWM_sources(prior_vector::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, source_length_limits::UnitRange{<:Integer})
                    srcvec = Vector{Tuple{Matrix{Float64},Int64}}()
                        for prior in prior_vector
                            if typeof(prior)<:AbstractVector{<:Dirichlet{<:AbstractFloat}}
                                if length(prior) < source_length_limits[1]
                                    length_dist=DiscreteNonParametric(
                                        [source_length_limits...],
                                        [PRIOR_LENGTH_MASS,ones(length(source_length_limits[2:end]))/length(source_length_limits[2:end]).*(1-PRIOR_LENGTH_MASS)])
                                    PWM_length=rand(length_dist)
                                elseif source_length_limits[1] < length(prior) < source_length_limits[end]
                                    length_dist=DiscreteNonParametric(
                                        [source_length_limits...],
                                        [
                                            (ones(length(prior)-source_length_limits[1]+1)/(length(prior)-source_length_limits[1]+1).*PRIOR_LENGTH_MASS)...,
                                            (ones(source_length_limits[end]-length(prior))/(source_length_limits[end]-length(prior)).*(1-PRIOR_LENGTH_MASS))...
                                        ]
                                    )
                                    PWM_length=rand(length_dist)
                                else
                                    PWM_length=rand(source_length_limits)
                                end

                                if PWM_length>length(prior)
                                    prior_coord=rand(-(PWM_length-length(prior)):1)
                                else
                                    prior_coord=rand(1:length(prior)-PWM_length+1)
                                end

                                PWM = zeros(PWM_length,4)

                                curr_pos=prior_coord
                                for pos in 1:PWM_length
                                    curr_pos < 1 || curr_pos > length(prior) ? dirichlet=Dirichlet(ones(4)/4) :
                                        dirichlet=prior[curr_pos]
                                    PWM[pos, :] = rand(dirichlet)
                                end
                                push!(srcvec, (log.(PWM), prior_coord)) #push the source PWM to the source vector with the prior coord idx to allow drawing from the appropriate prior dirichlets on permuting source length
                            elseif prior==false
                                PWM_length=rand(source_length_limits)
                                PWM=zeros(PWM_length,4)
                                dirichlet=Dirichlet(ones(4)/4)
                                for pos in 1:PWM_length
                                    PWM[pos,:] = rand(dirichlet)
                                end
                                push!(srcvec, (log.(PWM), 1))
                            else
                                throw(ArgumentError("Bad prior supplied for ICA_PWM_Model!"))
                            end
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