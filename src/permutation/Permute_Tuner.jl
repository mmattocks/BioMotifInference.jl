mutable struct Permute_Tuner
    functions::AbstractVector{<:Function}
    velocities::Matrix{Float64}
    successes::BitMatrix #(memoryxfunc)
    minimum_clamp::AbstractFloat
    weights::AbstractVector{<:AbstractFloat}
    tabular_display::DataFrame
end

function Permute_Tuner(instruction::Permute_Instruct, clamp::AbstractFloat)
    nfuncs=length(instruction.funcs)
    assert_tuner(instruction.funcs, clamp) 
    vels=Dict{Function,Vector{Float64}}()
    succs=Dict{Function,BitVector}()
    funcnames=Vector{String}()
    vels=ones(TUNING_MEMORY*instruction.func_limit,nfuncs)
    succs=trues(TUNING_MEMORY*instruction.func_limit,nfuncs)
    for (idx,func) in enumerate(instruction.funcs)
        length(instruction.args[idx])>0 ? (kwstr="(+$(length(instruction.args[idx]))kwa)") : (kwstr="")
        push!(funcnames, string(nameof(func),kwstr))
    end

    tabular_display=DataFrame("Function"=>funcnames, "Succeed"=>zeros(Int64,nfuncs), "Fail"=>zeros(Int64, nfuncs),"Velocity"=>ones(nfuncs), "Weights"=>instruction.weights)

    return Permute_Tuner(instruction.funcs,vels,succs,clamp,instruction.weights,tabular_display)
end
    
function assert_tuner(functions, clamp)
    !(1/length(functions)>=clamp>0) && throw(ArgumentError("Minimum function call probability for tuner must be a positive float and cannot exceed 1/number of functions to tune."))
end

function tune_weights!(tuner::Permute_Tuner, call_report::Vector{Tuple{Int64,Float64,Float64}})
    for call in call_report
        funcidx,time,distance=call
        distance!==-Inf && (tuner.velocities[:,funcidx]=update_velocity!(tuner.velocities[:,funcidx],time,distance)) #do not push velocity to array if it has -Inf probability (usu no new model found)
        if call===call_report[end]
            tuner.successes[:,funcidx]=update_sucvec!(tuner.successes[:,funcidx],true)
            tuner.tabular_display[funcidx,"Succeed"]+=1
        else
            tuner.successes[:,funcidx]=update_sucvec!(tuner.successes[:,funcidx],true)
            tuner.tabular_display[funcidx,"Fail"]+=1 
        end
    end
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
    mvels=[mean(t.velocities[:,n]) for n in 1:length(t.functions)]
    t.tabular_display[!,"Velocity"]=mvels

    any(i->i<0,mvels) && (mvels.+=-minimum(mvels)+1.) #to calculate weights, scale negative values into >1.
    pvec=[mvels[n]*(sum(t.successes[:,n])/length(t.successes[:,n])) for n in 1:length(t.functions)]
    pvec./=sum(pvec)
    any(i->i<t.minimum_clamp,pvec) && clamp_pvec(pvec,t.minimum_clamp)

    isprobvec(pvec) && (t.weights=pvec)
    t.tabular_display[!,"Weights"]=pvec
end
            function clamp_pvec(pvec, clamp)
                vals_to_clamp=findall(i->i<clamp, pvec)
                vals_to_deplete=findall(i->i>clamp, pvec)
                pvec[vals_to_clamp].=clamp
                pvec[vals_to_deplete].-=length(vals_to_clamp)*clamp/length(vals_to_deplete)
            end


function tune_instruction(tuner::Permute_Tuner, i::Permute_Instruct)
    return Permute_Instruct(i.funcs, tuner.weights, i.model_limit, i.func_limit, args=i.args)
end

function Base.show(io::IO, tuner::Permute_Tuner; progress=false)
    show(io, tuner.tabular_display, rowlabel=:I, summary=false)
    progress && return(size(tuner.tabular_display,1)+4)
end