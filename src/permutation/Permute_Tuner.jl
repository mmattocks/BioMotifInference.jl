mutable struct Permute_Tuner
    functions::Vector{Function}
    velocities::Dict{Function,Vector{AbstractFloat}}
    successes::Dict{Function,BitVector}
    minimum_clamp::AbstractFloat
    weights::Categorical
    tabular_display::DataFrame
end

function Permute_Tuner(instruction::Permute_Instruct, clamp::AbstractFloat)
    assert_tuner(instruction.funcs, clamp) 
    vels=Dict{Function,Vector{Float64}}()
    succs=Dict{Function,BitVector}()
    funcnames=Vector{String}()
    for func in instruction.funcs
        vels[func]=ones(TUNING_MEMORY)
        succs[func]=trues(TUNING_MEMORY)
        push!(funcnames, string(nameof(func)))
    end

    tabular_display=DataFrame("Function"=>funcnames, "Succeed"=>zeros(Int64,length(instruction.funcs)), "Fail"=>zeros(Int64, length(instruction.funcs)),"Velocity"=>ones(length(instruction.funcs)), "Weights"=>instruction.weights.p)

    return Permute_Tuner(instruction.funcs,vels,succs,clamp,instruction.weights,tabular_display)
end
    
function assert_tuner(functions, clamp)
    !(1/length(functions)>=clamp>0) && throw(ArgumentError("Minimum function call probability for tuner must be a positive float and cannot exceed 1/number of functions to tune."))
end

function tune_weights!(tuner::Permute_Tuner, call_report::Vector{Tuple{Function,Float64,Float64}})
    for call in call_report
        func,time,distance=call
        distance!==-Inf && (tuner.velocities[func]=update_velocity!(tuner.velocities[func],time,distance)) #do not push velocity to array if it has -Inf probability (usu no new model found)
        if call===call_report[end]
            update_sucvec!(tuner.successes[func],true)
            tuner.tabular_display[findfirst(isequal(string(nameof(func))),tuner.tabular_display.Function),"Succeed"]+=1
        else
            update_sucvec!(tuner.successes[func],false)
            tuner.tabular_display[findfirst(isequal(string(nameof(func))),tuner.tabular_display.Function),"Fail"]+=1 
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
    mvels=zeros(length(t.functions))

    for (n,func) in enumerate(t.functions)
        mvels[n]=mean(t.velocities[func])
        t.tabular_display[n,"Velocity"]=mvels[n]
    end

    any(i->i<0,mvels) && (mvels.+=-minimum(mvels)+1.) #to calculate weights, scale negative values into >1.
    pvec=[mvels[n]*(sum(t.successes[func])/length(t.successes[func])) for (n,func) in enumerate(t.functions)]
    pvec./=sum(pvec)
    any(i->i<t.minimum_clamp,pvec) && clamp_pvec(pvec,t.minimum_clamp)

    isprobvec(pvec) && (t.weights=Categorical(pvec))
    t.tabular_display[!,"Weights"]=pvec
end
            function clamp_pvec(pvec, clamp)
                vals_to_clamp=findall(i->i<clamp, pvec)
                vals_to_deplete=findall(i->i>clamp, pvec)
                pvec[vals_to_clamp].=clamp
                pvec[vals_to_deplete].-=length(vals_to_clamp)*clamp/length(vals_to_deplete)
            end


function tune_instruction(tuner::Permute_Tuner, i::Permute_Instruct)
    return Permute_Instruct(i.funcs, tuner.weights.p, i.model_limit, i.func_limit, args=i.args)
end

function Base.show(io::IO, tuner::Permute_Tuner; progress=false)
    show(io, tuner.tabular_display, rowlabel=:I, summary=false)
    progress && return(size(tuner.tabular_display,1)+4)
end