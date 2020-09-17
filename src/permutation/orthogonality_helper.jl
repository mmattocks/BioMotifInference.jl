##ORTHOGONALITY HELPER
function consolidate_srcs(con_idxs::Dict{Integer,Vector{Integer}}, m::ICA_PWM_Model, obs_array::AbstractMatrix{<:Integer}, obs_lengths::AbstractVector{<:Integer}, bg_scores::AbstractMatrix{<:AbstractFloat}, contour::AbstractFloat,  models::AbstractVector{<:Model_Record}; iterates::Integer=length(m.sources)*2, remote=false) 
    new_log_Li=-Inf;  iterate = 1
    T,O = size(obs_array); T=T-1; S = length(m.sources)
    new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
    a, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true)

    while new_log_Li <= contour && iterate <= iterates #until we produce a model more likely than the lh contour or exceed iterates
        new_sources=deepcopy(m.sources); new_mix=deepcopy(m.mix_matrix)
        clean=Vector{Bool}(trues(O))

        for host_src in filter(!in(vcat(values(con_idxs)...)), keys(con_idxs)) #copy mix information to the source to be consolidated on as host
            for cons_src in con_idxs[host_src]
                new_mix[:,host_src].=[new_mix[o,host_src] || new_mix[o,cons_src] for o in 1:size(new_mix,1)]
            end
        end

        remote ? (merger_m = deserialize(rand(models).path)) : (merger_m = remotecall_fetch(deserialize, 1, rand(models).path)) #randomly select a model to merge
        used_m_srcs=Vector{Int64}()

        for src in unique(vcat(values(con_idxs)...)) #replace all non-host sources with sources from a merger model unlike the one being removed
            distvec=[pwm_distance(m.sources[src][1],m_src[1]) for m_src in merger_m.sources]
            m_src=findmax(distvec)[2]
            while m_src in used_m_srcs
                distvec[m_src]=0.; m_src=findmax(distvec)[2]
                length(used_m_srcs)==length(merger_m.sources) && break;break
            end

            clean[new_mix[:,src]].=false #mark dirty any obs that start with the source
            new_sources[src]=merger_m.sources[m_src]
            new_mix[:,src]=merger_m.mix_matrix[:,m_src]
            clean[new_mix[:,src]].=false #mark dirty any obs that end with the source
            push!(used_m_srcs, m_src)
        end
        
        if consolidate_check(new_sources)[1] #if the new sources pass the consolidate check
            new_log_Li, cache = IPM_likelihood(new_sources, obs_array, obs_lengths, bg_scores, new_mix, true, true, cache, clean) #assess likelihood
        end

        iterate += 1
    end

    return ICA_PWM_Model("candidate","consolidated $(m.origin)", new_sources, m.source_length_limits, new_mix, new_log_Li)
end

function consolidate_check(sources::AbstractVector{<:Tuple{<:AbstractMatrix{<:AbstractFloat},<:Integer}}; thresh=CONSOLIDATE_THRESH, revcomp=REVCOMP)
    pass=true
    lengthδmat=[size(src1[1],1) - size(src2[1],1) for src1 in sources, src2 in sources]
    cons_idxs=Dict{Integer,Vector{Integer}}()
    for src1 in 1:length(sources), src2 in src1+1:length(sources)
        if lengthδmat[src1,src2]==0
            revcomp ? (info_condition = (pwm_distance(sources[src1][1],sources[src2][1]) < thresh || pwm_distance(sources[src1][1],revcomp_pwm(sources[src2][1])) < thresh)) : (info_condition = (pwm_distance(sources[src1][1],sources[src2][1]) < thresh))
            if info_condition
                if !in(src1,keys(cons_idxs))
                    cons_idxs[src1]=[src2]; pass=false
                else
                    push!(cons_idxs[src1], src2)
                end
            end
        end
    end

    return pass, cons_idxs
end

                function pwm_distance(pwm1,pwm2)
                    minwml=min(size(pwm1,1),size(pwm2,1))
                    return sum([euclidean(exp.(pwm1[pos,:]), exp.(pwm2[pos,:])) for pos in 1:minwml])/minwml
                end
