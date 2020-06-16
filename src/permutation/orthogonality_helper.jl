
##ORTHOGONALITY HELPER
function consolidate_srcs(con_idxs::Vector{Vector{Integer}}, sources::Vector{Tuple{Matrix{AbstractFloat},Integer}}, mix::BitMatrix, observations::Matrix{Integer}, obs_lengths::Vector{Integer}, bg_scores::Matrix{AbstractFloat}, source_priors::Vector{Vector{Dirichlet{AbstractFloat}}}, informed_sources::Integer, source_length_limits::UnitRange)
    new_sources=deepcopy(sources);new_mix=deepcopy(mix)
    consrc=0
    for (s,convec) in enumerate(con_idxs)
        if length(convec)>0
            consrc=s
            for cons in convec
                new_sources[cons]=init_logPWM_sources([source_priors[cons]], source_length_limits)[1]
                new_mix[:,s]=[any([new_mix[i,s],mix[i,cons]]) for i in 1:size(mix,1)] #consolidate the new_mix source with the consolidated mix
            end
            break #dont do this for more than one consolidation at a time
        end
    end

    return fit_mix(ICA_PWM_model("consolidate", new_sources, informed_sources, source_length_limits, new_mix, -Inf, [""]), observations, obs_lengths, bg_scores, consrc)
end

function consolidate_check(sources::Vector{Tuple{Matrix{AbstractFloat},Integer}}; thresh=.035)
    pass=true
    con_idxs=Vector{Vector{Integer}}()
    for (s1,src1) in enumerate(sources)
        s1_idxs=Vector{Integer}()
        for (s2,src2) in enumerate(sources)
            if !(s1==s2)
                pwm1=src1[1]; pwm2=src2[1]
                if -3<=(size(pwm1,1)-size(pwm2,1))<=3 
                    if pwm_distance(pwm1,pwm2)<thresh
                        push!(s1_idxs,s2)
                        pass=false
                    end
                end
            end
        end
        push!(con_idxs,s1_idxs)
    end
    return pass, con_idxs
end

                function pwm_distance(pwm1,pwm2)
                    minwml=min(size(pwm1,1),size(pwm2,1))
                    return sum([euclidean(exp.(pwm1[pos,:]), exp.(pwm2[pos,:])) for pos in 1:minwml])/minwml
                end
