
function dev_likelihood(sources::AbstractVector{<:Tuple{<:AbstractMatrix{<:AbstractFloat},<:Integer}}, observations::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractArray{<:AbstractFloat}, mix::BitMatrix, revcomp::Bool=true, returncache::Bool=false, cache::AbstractVector{<:AbstractFloat}=zeros(0), clean::AbstractVector{<:Bool}=Vector(falses(size(observations)[2])))
    source_wmls=[size(source[1])[1] for source in sources]
    O = size(bg_scores)[2]
    source_stops=[obsl-wml+1 for wml in source_wmls, obsl in obs_lengths] #stop scannng th source across the observation as the source reaches the end        
    L=maximum(obs_lengths)+1

    obs_src_idxs=mix_pull_idxs(mix) #get vectors of sources emitting in each obs

    revcomp ? (srcs=[cat(source[1],revcomp_pwm(source[1]),dims=3) for source in sources]; motif_expectations = [(0.5/obsl) for obsl in obs_lengths]; mat_dim=2) : (src=[source[1] for source in sources]; ; motif_expectations = [(1/obsl) for obsl in obs_lengths]; mat_dim=1) #setup appropriate reverse complemented sources if necessary and set log_motif_expectation-nMica has 0.5 per base for including the reverse complement, 1 otherwise

    lme_vec=zeros(length(sources))

    obs_lhs=Vector{Vector{Float64}}() #setup likelihood vecs for threaded operation-reassembled on return
    nt=Threads.nthreads()
    for t in 1:nt-1
        push!(obs_lhs,zeros(Int(floor(O/nt))))
    end
    push!(obs_lhs, zeros(Int(floor(O/nt)+(O%nt))))
    
    Threads.@threads for t in 1:nt
        weavevec=zeros(3)
        opt = floor(O/nt) #obs per thread
        score_mat=zeros(maximum(source_stops),mat_dim)
        score_matrices=Vector{Matrix{Float64}}(undef, length(sources)) #preallocate
        osi_emitting=Vector{Int64}() #preallocate
        lh_vec = zeros(L) #preallocated likelihood vector is one position (0 initialiser) longer than the longest obs

        for i in 1:Int(opt+(t==nt)*(O%nt))
            o=Int(i+(t-1)*opt)
            if clean[o]
                obs_lhs[t][i]=cache[o]
            else
                obsl = obs_lengths[o]
                oidxs=obs_src_idxs[o]
                mixwmls=source_wmls[oidxs]

                obs_cardinality = length(oidxs) #the more sources, the greater the cardinality_penalty
                if obs_cardinality > 0
                    d_score_obs_sources!(score_mat, score_matrices, view(srcs,oidxs), view(observations,1:obsl,o), obsl, mixwmls, view(source_stops,oidxs,o), revcomp)
                    lme_vec.=motif_expectations[o]
                    penalty_sum = sum(lme_vec[1:obs_cardinality])
                    penalty_sum > 1. && (penalty_sum=1.)
                    cardinality_penalty=log(1.0-penalty_sum)
                else
                    cardinality_penalty=0.0
                end

                obs_lhs[t][i]=d_weave_scores!(weavevec, lh_vec, obsl, view(bg_scores,:,o), score_matrices, oidxs, mixwmls, log(motif_expectations[o]), cardinality_penalty, osi_emitting, revcomp)

                empty!(osi_emitting)

            end
        end
    end

    returncache ? (return lps([lps(obs_lhs[t]) for t in 1:nt]), vcat(obs_lhs...)) : (return lps([lps(obs_lhs[t]) for t in 1:nt]))
end

    @inline function mix_pull_idxs(A::AbstractArray{Bool})
        n=count(A)
        S=[Vector{Int64}() for o in 1:size(A,2)]
        cnt=1
        for (i,a) in pairs(A)
            if a
                push!(S[i[2]],i[1])
                cnt+=1
            end
        end
        S
    end

    @inline function d_score_obs_sources!(score_mat, score_matrices::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, sources, observation::AbstractVector{<:Integer}, obsl::Integer, source_wmls::AbstractVector{<:Integer}, source_stops, revcomp=true) 
        for (s,source) in enumerate(sources)
            for t in 1:source_stops[s]
                for position in 1:size(source)[1]
                    score_loc = t+position-1 #score_loc is the position of the obs to be scored by PWM
                    score_mat[t,1] += source[position,observation[score_loc],1] #add the appropriate log PWM value from the source to the score
                    score_mat[t,2] += source[position,observation[score_loc],2] #add the appropriate log PWM value from the source to the score

                end
            end
            score_matrices[s]=score_mat[1:source_stops[s],:] #copy score matrix to vector
            score_mat.=0. #reset score matrix
        end
    end


    @inline function d_weave_scores!(weavevec, lh_vec, obsl::Integer, bg_scores::SubArray, score_mat::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, obs_source_indices::AbstractVector{<:Integer}, source_wmls::AbstractVector{<:Integer}, log_motif_expectation::AbstractFloat, cardinality_penalty::AbstractFloat, osi_emitting, revcomp::Bool)
        for i in 2:obsl+1 #i=1 is ithe lh_vec initializing 0, i=2 is the score of the first background position (ie t=1)
            t=i-1
            score = lps(lh_vec[i-1], bg_scores[t], cardinality_penalty)
    
            #logic: all observations are scored from t=wml to the end of the obs-therefore check at each position for new sources to add (indexed by vector position to retrieve source wml and score matrix)
            if length(osi_emitting)<length(obs_source_indices)
                for n in 1:length(obs_source_indices)
                    if !(n in osi_emitting)
                        t>= source_wmls[n] && (push!(osi_emitting,n))
                    end
                end
            end
    
            for n in osi_emitting
                wml = source_wmls[n]
                from_score = lh_vec[i-wml+1] #score at the first position of the PWM
                score_array = score_mat[n] #get the source score matrix
                score_idx = t - wml + 1 #translate t to score_array idx for emission score
                f_emit_score = score_array[score_idx,1] #emission score at the last position of the PWM
                r_emit_score = score_array[score_idx,2]

                weavevec .= score, lps(from_score, f_emit_score, log_motif_expectation), lps(from_score, r_emit_score, log_motif_expectation)

                score=logsumexp(weavevec)

                #score = logsumexp([score, lps(from_score, f_emit_score, log_motif_expectation), lps(from_score, r_emit_score, log_motif_expectation)])

            end
            lh_vec[i] = score
        end
        return lh_vec[obsl+1]
    end


    # source_wmls=[size(source[1])[1] for source in sources]
    # source_stops=[obsl-wml+1 for wml in source_wmls, obsl in obs_lengths] #stop scannng th source across the observation as the source reaches the end
    
    # O = size(bg_scores)[2]
    # nt=Threads.nthreads()
    # obs_lhs=[zeros(Int(floor(O/nt))) for t in 1:nt-1]
    # push!(obs_lhs, zeros(Int(floor(O/nt)+(O%nt))))
    
    # Threads.@threads for t in 1:nt
    #     opt = floor(O/nt) #obs per thread
    #     osi_emitting=Vector{Int64}() #preallocate mutable struct to be pushed to whilst tracking emitter indices
    #     for i in 1:Int(opt+(t==nt)*(O%nt))
    #         o=Int(i+(t-1)*opt)
            
    #         if clean[o]
    #             obs_lhs[t][i]=cache[o]
    #         else
    #             obsl = obs_lengths[o]
    #             mixview=findall(view(mix,o,:))
    #             obs_cardinality = length(mixview) #the more sources, the greater the cardinality_penalty
    #             lme=log_motif_expectations[o]

    #             if obs_cardinality > 0 
    #                 mixwmls=source_wmls[mixview]
    #                 stopview=view(source_stops,mixview,o)    
    #                 score_mat=zeros(maximum(stopview),obs_cardinality,mat_dim)
    #                 revcomp ? (score_sources_ds!(score_mat, srcs[mixview], view(observations,1:obsl,o), mixwmls, stopview)) : (score_source_ss!(score_mat, sources, observation, source_stops[s])) #get the scores for this obs
    #                 penalty_sum = sum(exp.(fill(lme,obs_cardinality)))
    #                 penalty_sum > 1. && (penalty_sum=1.)
    #                 cardinality_penalty=log(1.0-penalty_sum)
    #             else
    #                 cardinality_penalty=0.0
    #                 score_mat=zeros(0,0,0)
    #                 mixwmls=zeros(Int64,0)
    #             end

    #             obs_lhs[t][i]=dev_weave(obsl, view(bg_scores,:,o), score_mat, obs_cardinality, mixwmls, lme, cardinality_penalty, osi_emitting, revcomp)
    #             empty!(osi_emitting)
    #         end
    #     end
    # end

                function score_sources_ds!(score_matrix, sources, observation, wmls, source_stops, revcomp::Bool=true)
                    wml_offsets=[wml-minimum(wmls) for wml in wmls]
                    
                    for (s, source) in enumerate(sources)
                        for t in 1:source_stops[s]
                            for position in 1:size(source)[1]
                                score_loc = t+position-1 #score_loc is the position of the obs to be scored by PWM
                                score_matrix[t,s,1] += source[position,observation[score_loc],1] #add the appropriate log PWM value from the source to the score
                                score_matrix[t,s,2] += source[position,observation[score_loc],2]
                            end
                        end
                        
                        return score_matrix
                    end
                end

                function score_source_ss(observation::AbstractVector{<:Integer}, source::AbstractMatrix{<:AbstractFloat}, source_stop::Integer, revcomp::Bool=true)
                    score_vec = zeros(source_stop,1)

                    for t in 1:source_stop
                        forward_score = 0.0;  #initialise scores as log(p=1)

                        for position in 1:size(source)[1]
                            score_loc = t+position-1 #score_loc is the position of the obs to be scored by PWM
                            forward_score += source[position,observation[score_loc]] #add the appropriate log PWM value from the source to the score
                        end

                        score_vec[t,1] = forward_score #assign scores to the matrix
                    end
                    
                    return score_vec
                end

                function dev_weave(obsl::Integer, bg_scores::SubArray, score_mat::AbstractArray{<:AbstractFloat}, no_srcs::Integer, source_wmls::AbstractVector{<:Integer}, log_motif_expectation::AbstractFloat, cardinality_penalty::AbstractFloat,  osi_emitting::Vector{Int64}, revcomp::Bool=true)
                    L=obsl+1
                    lh_vec = zeros(L)#likelihood vector is one position (0 initialiser) longer than the observation
                    revcomp ? ds_weave!(score_mat, L, lh_vec, bg_scores, cardinality_penalty, osi_emitting, no_srcs, source_wmls, log_motif_expectation) : ss_weave!(score_mat, L, lh_vec, bg_scores, cardinality_penalty, osi_emitting, no_srcs, source_wmls, log_motif_expectation)

                    return lh_vec[end]
                end

                function ds_weave!(score_mat, L::Integer, lh_vec::Vector{Float64}, bg_scores::SubArray, cardinality_penalty::Float64, osi_emitting::Vector{Int64}, no_srcs::Integer, source_wmls::Vector{<:Integer}, log_motif_expectation::Float64)
                    length(source_wmls) > 0 && (minwml=minimum(source_wmls))
                    scorevec=zeros(3)
                    for i in 2:L #i=1 is ithe lh_vec initializing 0, i=2 is the score of the first background position (ie t=1)
                        t=i-1
                        score = lps.(view(lh_vec,t), view(bg_scores,t), cardinality_penalty)
                
                        #logic: all observations are scored from t=wml to the end of the obs-therefore check at each position for new sources to add (indexed by vector position to retrieve source wml and score matrix)
                        if length(osi_emitting)<no_srcs
                            for n in 1:no_srcs
                                if !(n in osi_emitting)
                                    t>=source_wmls[n] && (push!(osi_emitting,n))
                                end
                            end
                        end
                
                        for n in osi_emitting
                            wml = source_wmls[n]
                            from_score = lh_vec[i-wml+1] #score at the first position of the PWM
                            score_idx = t - minwml + 1 #translate t to score_array idx for emission score
                            f_emit_score = score_mat[score_idx,n,1] #emission score at the last position of the PWM
                            r_emit_score = score_mat[score_idx,n,2]
                            scorevec[1:3].=score, lps(from_score, f_emit_score, log_motif_expectation), lps(from_score, r_emit_score, log_motif_expectation)
                            score = logsumexp(scorevec)
                        end
                        lh_vec[i] = score
                    end
                end

                function ss_weave!(score_mat, L, lh_vec, bg_scores, cardinality_penalty, osi_emitting, no_srcs, source_wmls, log_motif_expectation)
                    for i in 2:L #i=1 is ithe lh_vec initializing 0, i=2 is the score of the first background position (ie t=1)
                        t=i-1
                        score = lps(lh_vec[i-1], bg_scores[t], cardinality_penalty)
                
                        #logic: all observations are scored from t=wml to the end of the obs-therefore check at each position for new sources to add (indexed by vector position to retrieve source wml and score matrix)
                        if length(osi_emitting)<no_srcs
                            for n in 1:no_srcs
                                if !(n in osi_emitting)
                                    t>= source_wmls[n] && (push!(osi_emitting,n))
                                end
                            end
                        end
                
                        for n in osi_emitting
                            wml = source_wmls[n]
                            from_score = lh_vec[i-wml+1] #score at the first position of the PWM
                            score_array = score_mat[n] #get the source score matrix
                            score_idx = t - wml + 1 #translate t to score_array idx for emission score
                            emit_score = score_array[score_idx,1] #emission score at the last position of the PWM
                            score = logaddexp(score, lps(from_score, emit_score, log_motif_expectation))
                        end
                        lh_vec[i] = score
                    end
                end