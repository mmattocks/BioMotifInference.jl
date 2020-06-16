
##BASIC UTILITY FUNCTIONS
#SOURCE PERMUTATION
function permute_source_weights(source::Tuple{Matrix{AbstractFloat},Integer}, shift_freq::AbstractFloat, PWM_shift_dist::Distributions.ContinuousUnivariateDistribution)
    dirty=false; source_length=size(source[1],1)
    new_source=deepcopy(source)

    for source_pos in 1:source_length
        if rand() <= shift_freq
            pos_WM = exp.(source[1][source_pos,:]) #leaving logspace, get the wm at that position
            new_source[1][source_pos,:] = log.(wm_shift(pos_WM, PWM_shift_dist)) #accumulate probabilty at a randomly selected base, reassign in logspace and carry on
            !dirty && (dirty=true)
        end
    end

    if !dirty #if no positions were shifted, pick one and shift
        rand_pos=rand(1:source_length)
        pos_WM = exp.(source[1][rand_pos,:])
        new_source[1][rand_pos,:]=log.(wm_shift(pos_WM, PWM_shift_dist))
    end

    return new_source
end

                function wm_shift(pos_WM::Vector{AbstractFloat}, PWM_shift_dist::Distributions.ContinuousUnivariateDistribution)
                    base_to_shift = rand(1:4) #pick a base to accumulate probability
                    permute_sign = rand(-1:2:1)
                    shift_size = rand(PWM_shift_dist)
                    new_wm=zeros(4)
                    
                    for base in 1:4 #ACGT
                        if base == base_to_shift
                            new_wm[base] =
                            clamp(0, #no lower than 0 prob
                            (pos_WM[base]          #selected PWM posn
                            + permute_sign * shift_size), #randomly permuted by size param
                            1) #no higher than prob 1
                        else
                            size_frac = shift_size / 3 #other bases shifted in the opposite direction by 1/3 the shift accumulated at the base to permute
                            new_wm[base] =
                            clamp(0,
                            (pos_WM[base]
                            - permute_sign * size_frac),
                            1)
                        end
                    end
                    new_wm = new_wm ./ sum(new_wm) #renormalise to sum 1 - necessary in case of clamping at 0 or 1
                    !isprobvec(new_wm) && throw(DomainError(new_wm, "Bad weight vector generated in wm_shift!")) #throw assertion exception if the position WM is invalid
                    return new_wm
                end



function permute_source_length(source::Tuple{Matrix{AbstractFloat},Integer}, prior::Vector{Dirichlet{AbstractFloat}}, length_limits::UnitRange{Integer}, permute_range::UnitRange{Integer}=1:3, uninformative::Dirichlet=Dirichlet([.25,.25,.25,.25]))
    source_PWM, prior_idx = source
    source_length = size(source_PWM,1)

    permute_sign, permute_length = get_length_params(source_length, length_limits, permute_range)

    permute_sign==1 ? permute_pos = rand(1:source_length+1) :
        permute_pos=rand(1:source_length-permute_length)
    
    if permute_sign == 1 #if we're to add positions to the PWM
        ins_WM=zeros(permute_length,4)
        for pos in 1:permute_length
            prior_position=permute_pos+prior_idx
            prior_position<1 || prior_position>length(prior) ? 
                ins_WM[pos,:] = log.(transpose(rand(uninformative))) :
                ins_WM[pos,:] = log.(transpose(rand(prior[prior_position])))
                !isprobvec(exp.(ins_WM[pos,:])) && throw(DomainError(ins_WM, "Bad weight vector generated in permute_source_length!"))
        end
        upstream_source=source_PWM[1:permute_pos-1,:]
        downstream_source=source_PWM[permute_pos:end,:]
        source_PWM=vcat(upstream_source,ins_WM,downstream_source)
        permute_pos==1 && (prior_idx-=permute_length)
    else #if we're to remove positions
        upstream_source=source_PWM[1:permute_pos-1,:]
        downstream_source=source_PWM[permute_pos+permute_length:end,:]
        source_PWM=vcat(upstream_source,downstream_source)
        permute_pos==1 && (prior_idx+=permute_length)
    end

    return (source_PWM, prior_idx) #return a new source
end

                function get_length_params(source_length::Integer, length_limits::UnitRange{Integer}, permute_range::UnitRange{Integer})
                    extendable = length_limits[end]-source_length
                    contractable =  source_length-length_limits[1]

                    if extendable == 0 && contractable > 0
                        permute_sign=-1
                    elseif contractable == 0 && extendable > 0
                        permute_sign=1
                    else
                        permute_sign = rand(-1:2:1)
                    end

                    permute_sign==1 && extendable<permute_range[end] && (permute_range=permute_range[1]:extendable)
                    permute_sign==-1 && contractable<permute_range[end] && (permute_range=permute_range[1]:contractable)
                    permute_length = rand(permute_range)

                    return permute_sign, permute_length
                end

function erode_source(source::Tuple{Matrix{AbstractFloat},Integer},length_limits::UnitRange{Integer},info_thresh)
    pwm,prior_idx=source
    infovec=get_pwm_info(pwm)
    start_idx,end_idx=get_erosion_idxs(infovec, info_thresh, length_limits)

    return new_source=(pwm[start_idx:end_idx,:], prior_idx+start_idx-1)
end

    function get_pwm_info(pwm::Matrix{AbstractFloat}; logsw::Bool=true)
        wml=size(pwm,1)
        infovec=zeros(wml)
        for pos in 1:wml
            logsw ? wvec=deepcopy(exp.(pwm[pos,:])) : wvec=deepcopy(pwm[pos,:])
            !isprobvec(wvec) && throw(DomainError(wvec, "Bad wvec in get_pwm_info -Original sources must be in logspace!!"))
            wvec.+=10^-99
            infscore = (2.0 + sum([x*log(2,x) for x in wvec]))
            infovec[pos]=infscore
        end
        return infovec
    end

    function get_erosion_idxs(infovec::Vector{AbstractFloat}, info_thresh::AbstractFloat, length_limits::UnitRange{Integer})
        srcl=length(infovec)
        contractable =  srcl-length_limits[1]
        contractable <=0 && throw(DomainError(contractable, "erode_source passed a source at its lower length limit!"))
        centeridx=findmax(infovec)[2]
        
        start_idx=findprev(info-><(info,info_thresh),infovec,centeridx)
        start_idx===nothing ? (start_idx=1) : (start_idx+=1)
        end_idx=findnext(info-><(info, info_thresh),infovec,centeridx)
        end_idx===nothing ? (end_idx=srcl) : (end_idx-=1)

        pos_to_erode=srcl-(end_idx-start_idx)
        if pos_to_erode > contractable
            pos_to_restore = pos_to_erode-contractable
            while pos_to_restore>0
                end_die=rand()
                if end_die <= .5
                    start_idx>1 && (pos_to_restore-=1; start_idx-=1)
                else
                    end_idx<srcl && (pos_to_restore-=1; end_idx+=1)
                end
            end
        end

        return start_idx, end_idx
    end

#MIX MATRIX FUNCTIONS
function mixvec_decorrelate(mix::BitVector, moves::Integer)
    new_mix=deepcopy(mix)
    idxs_to_flip=rand(1:length(mix), moves)
    new_mix[idxs_to_flip] .= .!mix[idxs_to_flip]
    return new_mix
end

function mix_matrix_decorrelate(mix::BitMatrix, moves::Integer)
    clean=Vector{Bool}(trues(size(mix,1)))
    new_mix=deepcopy(mix)
    indices_to_flip = rand(CartesianIndices(mix), moves)
    new_mix[indices_to_flip] .= .!mix[indices_to_flip]
    clean[unique([idx[1] for idx in indices_to_flip])] .= false #mark all obs that had flipped indices dirty
    return new_mix, clean
end


function most_dissimilar(mix1, mix2)
    S1=size(mix1,2);S2=size(mix2,2)
    dist_mat=zeros(S1,S2)
    for s1 in 1:S1, s2 in 1:S2
        dist_mat[s1,s2]=sum(mix1[:,s1].==mix2[:,s2])
    end
    scores=vec(sum(dist_mat,dims=1))
    return findmin(scores)[2]
end

function most_similar(src_mixvec, target_mixmat)
    src_sim = [sum(src_mixvec.==target_mixmat[:,s]) for s in 1:size(target_mixmat,2)] #compose array of elementwise equality comparisons between mixvectors and sum to score
    merge_s=findmax(src_sim)[2] #source from merger model will be the one with the highest equality comparison score
end