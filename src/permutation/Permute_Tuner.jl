mutable struct Permute_Tuner
    inst::Permute_Instruct
    velocities::Matrix{Float64}
    successes::BitMatrix #(memoryxfunc)
    tabular_display::DataFrame
    time_history::Vector{Float64}
    override::Bool
end

"""
    Permute_Tuner(instruction)

Generate a Permute_Tuner for the given instruction.
"""
function Permute_Tuner(instruction::Permute_Instruct)
    nfuncs=length(instruction.funcs)
    funcnames=Vector{String}()
    vels=ones(TUNING_MEMORY*instruction.func_limit,nfuncs)
    succs=trues(TUNING_MEMORY*instruction.func_limit,nfuncs)
    for (idx,func) in enumerate(instruction.funcs)
        length(instruction.args[idx])>0 ? (kwstr="(+$(length(instruction.args[idx]))kwa)") : (kwstr="")
        push!(funcnames, string(nameof(func),kwstr))
    end

    tabular_display=DataFrame("Function"=>funcnames, "Succeed"=>zeros(Int64,nfuncs), "Fail"=>zeros(Int64, nfuncs),"Velocity"=>ones(nfuncs), "Weights"=>instruction.weights)

    instruction.override_time>0. ? (override=true) : (override=false)
    return Permute_Tuner(instruction,vels,succs,tabular_display,zeros(CONVERGENCE_MEMORY),override)
end

"""
    tune_weights!(tuner, call_report)

Given a call_report from permute_IPM(), adjust tuner's Permute_Instruct weights on the basis of function success and likelihood surface velocity.
"""
function tune_weights!(tuner::Permute_Tuner, call_report::Vector{Tuple{Int64,Float64,Float64}})
    for call in call_report
        funcidx,time,distance=call
        distance!==-Inf && (tuner.velocities[:,funcidx]=update_velocity!(tuner.velocities[:,funcidx],time,distance)) #do not push velocity to array if it has -Inf probability (usu no new model found)
        if call===call_report[end]
            tuner.successes[:,funcidx]=update_sucvec!(tuner.successes[:,funcidx],true)
            tuner.tabular_display[funcidx,"Succeed"]+=1
        else
            tuner.successes[:,funcidx]=update_sucvec!(tuner.successes[:,funcidx],false)
            tuner.tabular_display[funcidx,"Fail"]+=1 
        end
    end
    tuner.override && mean(tuner.time_history)>tuner.inst.override_time ? (tuner.inst.weights=tuner.inst.override_weights; tuner.tabular_display[!,"Weights"]=tuner.inst.override_weights) :
        update_weights!(tuner)
end

function update_velocity!(velvec,time,distance)
    popfirst!(velvec) #remove first value
    vel = distance - log(time)
    push!(velvec,distance-log(time))
end

function update_sucvec!(sucvec, bool)
    popfirst!(sucvec)
    push!(sucvec, bool)
end

function update_weights!(t::Permute_Tuner)
    mvels=[mean(t.velocities[:,n]) for n in 1:length(t.inst.funcs)]
    t.tabular_display[!,"Velocity"]=copy(mvels)

    any(i->i<0,mvels) && (mvels.+=-minimum(mvels)+1.) #to calculate weights, scale negative values into >1.
    pvec=[mvels[n]*(sum(t.successes[:,n])/length(t.successes[:,n])) for n in 1:length(t.inst.funcs)]
    pvec./=sum(pvec)
    (any(pvec.<t.inst.min_clmps) || any(pvec.>t.inst.max_clmps)) && clamp_pvec!(pvec,t.inst.min_clmps,t.inst.max_clmps)

    @assert isprobvec(pvec)
    t.inst.weights=pvec; t.tabular_display[!,"Weights"]=pvec
end

"""
    clamp_pvec(pvec, tuner)

Clamp the values of a probability vector between the minimums and maximums provided by a Permute_Tuner.
"""
            function clamp_pvec!(pvec, min_clmps, max_clmps)
                #logic- first accumulate on low values, then distribute excess from high values
                any(pvec.<min_clmps) ? (low_clamped=false) : (low_clamped=true)
                while !low_clamped
                    vals_to_accumulate=pvec.<min_clmps
                    vals_to_deplete=pvec.>min_clmps

                    depletion=sum(min_clmps[vals_to_accumulate].-pvec[vals_to_accumulate])
                    pvec[vals_to_accumulate].=min_clmps[vals_to_accumulate]
                    pvec[vals_to_deplete].-=depletion/sum(vals_to_deplete)

                    !any(pvec.<min_clmps)&&(low_clamped=true)
                end

                any(pvec.>max_clmps) ? (high_clamped=false) : (high_clamped=true)
                while !high_clamped
                    vals_to_deplete=pvec.>max_clmps
                    vals_to_accumulate=pvec.<max_clmps

                    depletion=sum(pvec[vals_to_deplete].-max_clmps[vals_to_deplete])
                    pvec[vals_to_deplete].=max_clmps[vals_to_deplete]
                    pvec[vals_to_accumulate].+=depletion/sum(vals_to_accumulate)

                    !any(pvec.>max_clmps)&&(high_clamped=true)
                end
            end

function Base.show(io::IO, tuner::Permute_Tuner; progress=false)
    show(io, tuner.tabular_display, rowlabel=:I, summary=false)
    progress && return(size(tuner.tabular_display,1)+4)
end