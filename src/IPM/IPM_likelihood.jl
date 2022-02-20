#LIKELIHOOD SCORING FUNCS
function IPM_likelihood(sources::AbstractVector{<:Tuple{<:AbstractMatrix{<:AbstractFloat},<:Integer}}, observations::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractArray{<:AbstractFloat}, mix::BitMatrix, revcomp::Bool=REVCOMP, returncache::Bool=false, cache::AbstractVector{<:AbstractFloat}=zeros(0), clean::AbstractVector{<:Bool}=Vector(falses(size(observations)[2])))
    source_wmls=[size(source[1])[1] for source in sources]
    O = size(bg_scores)[2]
    source_stops=[obsl-wml+1 for wml in source_wmls, obsl in obs_lengths] #stop scannng th source across the observation as the source reaches the end        
    L=maximum(obs_lengths)+1

    obs_src_idxs=mix_pull_idxs(mix) #get vectors of sources emitting in each obs

    revcomp ? (srcs=[cat(source[1],revcomp_pwm(source[1]),dims=3) for source in sources]; motif_expectations = [((MOTIF_EXPECT/2)/obsl) for obsl in obs_lengths]; mat_dim=2) : (srcs=[source[1] for source in sources]; ; motif_expectations = [(MOTIF_EXPECT/obsl) for obsl in obs_lengths]; mat_dim=1) #setup appropriate reverse complemented sources if necessary and set log_motif_expectation-nMica has 0.5 per obs for including the reverse complement, 1 otherwise

    lme_vec=zeros(length(sources))

    obs_lhs=Vector{Vector{Float64}}() #setup likelihood vecs for threaded operation-reassembled on return
    nt=Threads.nthreads()
    for t in 1:nt-1
        push!(obs_lhs,zeros(Int(floor(O/nt))))
    end
    push!(obs_lhs, zeros(Int(floor(O/nt)+(O%nt))))
    
    Threads.@threads for t in 1:nt
        revcomp && (weavevec=zeros(3))
        revcomp ? (score_mat=zeros(maximum(source_stops),2)) : (score_mat=zeros(maximum(source_stops)))
        opt = floor(O/nt) #obs per thread
        score_matrices=Vector{typeof(score_mat)}(undef, length(sources)) #preallocate
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
                    revcomp ? score_sources_ds!(score_mat, score_matrices, view(srcs,oidxs), view(observations,:,o), view(source_stops,oidxs,o)) :
                         score_sources_ss!(score_mat, score_matrices, view(srcs,oidxs), view(observations,:,o), view(source_stops,oidxs,o)) #get scores for this observation


                    lme_vec.=motif_expectations[o]
                    penalty_sum = sum(lme_vec[1:obs_cardinality])
                    penalty_sum > 1. && (penalty_sum=1.)
                    cardinality_penalty=log(1.0-penalty_sum)
                else
                    cardinality_penalty=0.0
                end

                revcomp ? (obs_lhs[t][i]=weave_scores_ds!(weavevec, lh_vec, obsl, view(bg_scores,:,o), score_matrices, oidxs, mixwmls, log(motif_expectations[o]), cardinality_penalty, osi_emitting)) :
                    (obs_lhs[t][i]=weave_scores_ss!(lh_vec, obsl, view(bg_scores,:,o), score_matrices, oidxs, mixwmls, log(motif_expectations[o]), cardinality_penalty, osi_emitting))

                empty!(osi_emitting)

            end
        end
    end

    returncache ? (return lps([lps(obs_lhs[t]) for t in 1:nt]), vcat(obs_lhs...)) : (return lps([lps(obs_lhs[t]) for t in 1:nt]))
end
                @inline function mix_pull_idxs(A::AbstractArray{Bool})
                    n=count(A)
                    S=[Vector{Int64}() for o in 1:size(A,1)]
                    cnt=1
                    for (i,a) in pairs(A)
                        if a
                            push!(S[i[1]],i[2])
                            cnt+=1
                        end
                    end
                    return S
                end

                @inline function revcomp_pwm(pwm::AbstractMatrix{<:AbstractFloat}) #in order to find a motif on the reverse strand, we scan the forward strand with the reverse complement of the pwm, reordered 3' to 5', so that eg. an PWM for an ATG motif would become one for a CAT motif
                    return pwm[end:-1:1,end:-1:1]
                end

                @inline function score_sources_ds!(score_mat, score_matrices::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, sources, observation::AbstractVector{<:Integer}, source_stops) 
                    for (s,source) in enumerate(sources)
                        for t in 1:source_stops[s]
                            for position in 1:size(source,1)
                                score_loc = t+position-1 #score_loc is the position of the obs to be scored by PWM
                                score_mat[t,1] += source[position,observation[score_loc],1] #add the appropriate log PWM value from the source to the score
                                score_mat[t,2] += source[position,observation[score_loc],2] #add the appropriate log PWM value from the source to the score
            
                            end
                        end
                        score_matrices[s]=score_mat[1:source_stops[s],:] #copy score matrix to vector
                        score_mat.=0. #reset score matrix
                    end
                end

                @inline function score_sources_ss!(score_mat, score_matrices::AbstractVector{<:AbstractArray{<:AbstractFloat}}, sources, observation::AbstractVector{<:Integer}, source_stops) 
                    for (s,source) in enumerate(sources)
                        for t in 1:source_stops[s]
                            for position in 1:size(source,1)
                                score_loc = t+position-1 #score_loc is the position of the obs to be scored by PWM
                                score_mat[t] += source[position,observation[score_loc]] #add the appropriate log PWM value from the source to the score
            
                            end
                        end
                        score_matrices[s]=score_mat[1:source_stops[s]] #copy score matrix to vector
                        score_mat.=0. #reset score matrix
                    end
                end

                @inline function weave_scores_ds!(weavevec, lh_vec, obsl::Integer, bg_scores::SubArray, score_mat::AbstractVector{<:AbstractMatrix{<:AbstractFloat}}, obs_source_indices::AbstractVector{<:Integer}, source_wmls::AbstractVector{<:Integer}, log_motif_expectation::AbstractFloat, cardinality_penalty::AbstractFloat, osi_emitting)
                    for i in 2:obsl+1 #i=1 is the lh_vec initializing 0, i=2 is the score of the first background position (ie t=1)
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
                        end
                        lh_vec[i] = score
                    end
                    return lh_vec[obsl+1]
                end           

                @inline function weave_scores_ss!(lh_vec, obsl::Integer, bg_scores::SubArray, score_mat::AbstractVector{<:AbstractVector{<:AbstractFloat}}, obs_source_indices::AbstractVector{<:Integer}, source_wmls::AbstractVector{<:Integer}, log_motif_expectation::AbstractFloat, cardinality_penalty::AbstractFloat, osi_emitting)
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
                            f_emit_score = score_array[score_idx] #emission score at the last position of the PWM
            
                            score=logaddexp(score, lps(from_score, f_emit_score, log_motif_expectation))
                        end
                        lh_vec[i] = score
                    end
                    return lh_vec[obsl+1]
                end            