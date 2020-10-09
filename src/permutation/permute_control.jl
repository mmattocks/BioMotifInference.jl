mutable struct Permute_Instruct
    funcs::AbstractVector{<:Function}
    weights::AbstractVector{<:AbstractFloat}
    args::AbstractVector{<:AbstractVector{<:Tuple{<:Symbol,<:Any}}}
    model_limit::Integer
    func_limit::Integer
    min_clmps::AbstractVector{<:AbstractFloat}
    max_clmps::AbstractVector{<:AbstractFloat}
    override_time::AbstractFloat
    override_weights::AbstractVector{<:AbstractFloat}
    Permute_Instruct(funcs,
                    weights,
                    model_limit,
                    func_limit;
                    min_clmps=fill(.01,length(funcs)),
                    max_clmps=fill(1.,length(funcs)),
                    override_time=0., 
                    override_weights=zeros(length(funcs)),
                    args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcs)])=assert_permute_instruct(funcs,weights,args,model_limit,func_limit,min_clmps,max_clmps,override_time,override_weights) && new(funcs,weights,args,model_limit,func_limit,min_clmps,max_clmps,override_time,override_weights)
end

function assert_permute_instruct(funcs,weights,args,model_limit,func_limit,min_clmps, max_clmps, override_time, override_weights)
    !(length(funcs)==length(args)==length(weights)==length(min_clmps)==length(max_clmps)) && throw(ArgumentError("A valid Permute_Instruct must have as many tuning weights and argument vectors as functions!"))
    model_limit<1 && throw(ArgumentError("Permute_Instruct limit on models to permute must be positive Integer!"))
    func_limit<1 && throw(ArgumentError("Permute_Instruct limit on fuction calls per model permtued must be positive Integer!"))
    any(min_clmps.>=max_clmps) && throw(ArgumentError("Permute_Instruct max_clmps must all be greater than corresponding min_clmps!"))
    sum(min_clmps)>1. && throw(ArgumentError("Sum of minimum clamps must be <1.0 to maintain a valid probability vector!"))
    any(max_clmps.>1.) && throw(ArgumentError("Permute_Tuner maximum clamps cannot be >1.0!"))
    override_time<0. && throw(ArgumentError("Permute_Instruct override_time must be 0 (disabled) or positive float!"))
    override_time>0. && !isprobvec(override_weights) && throw(ArgumentError("Permute_Instruct override_weights must be valid probvec!"))
    return true
end


#permutation routine function- 
#general logic: receive array of permutation parameters, until a model more likely than the least is found:
#randomly select a model from the ensemble (the least likely having been removed by this point), then sample new models by permuting with each of hte given parameter sets until a model more likely than the current contour is found
#if none is found for the candidate model, move on to another candidate until the models_to_permute iterate is reached, after which return nothing for an error code

#four permutation modes: source (iterative random changes to sources until model lh>contour or iterate limit reached)
#						-(iterates, weight shift freq per source base, length change freq per source, weight_shift_dist (a ContinuousUnivariateDistribution)) for permute params
#						mix (iterative random changes to mix matrix as above)
#						-(iterates, unitrange of # of moves)
#						init (iteratively reinitialize sources from priors)
#						-(iterates) for init params
#						merge (iteratively copy a source + mix matrix row from another model in the ensemble until lh>contour or iterate						limit reached)
#						-(iterates) for merpge params


function permute_IPM(e::IPM_Ensemble, instruction::Permute_Instruct)
    call_report=Vector{Tuple{Int64,Float64,Float64}}()
	for model = 1:instruction.model_limit
		m_record = rand(e.models)
        m = deserialize(m_record.path)

        model_blacklist=copy(m.permute_blacklist)
        filteridxs, filtered_funcs, filtered_weights, filtered_args = filter_permutes(instruction, model_blacklist)

        for call in 1:instruction.func_limit
            start=time()
            funcidx=rand(filtered_weights)
            permute_func=filtered_funcs[funcidx]
            pos_args,kw_args=get_permfunc_args(permute_func,e,m,filtered_args[funcidx])
            new_m=permute_func(pos_args...;kw_args...)
            push!(call_report,(filteridxs[funcidx],time()-start,new_m.log_Li - e.contour))

            if length(new_m.permute_blacklist) > 0 && new_m.log_Li == -Inf #in this case the blacklisted function will not work on this model at all; it should be removed from the functions to use
                vcat(model_blacklist,new_m.permute_blacklist)
                filteridxs, filtered_funcs, filtered_weights, filtered_args = filter_permutes(instruction, model_blacklist)
            end

			dupecheck(new_m,m) && new_m.log_Li > e.contour && return new_m, call_report
		end
	end
	return nothing, call_report
end

function permute_IPM(e::IPM_Ensemble, job_chan::RemoteChannel, models_chan::RemoteChannel, comms_chan::RemoteChannel) #ensemble.models is partially updated on the worker to populate arguments for permute funcs
	persist=true
    id=myid()
    put!(comms_chan,id)
    while persist
        wait(job_chan)
        start=time()
        e.models, e.contour, instruction = fetch(job_chan)
        instruction == "stop" && (persist=false; break)

        call_report=Vector{Tuple{Int64,Float64,Float64}}()
        for model=1:instruction.model_limit
			found::Bool=false
            m_record = rand(e.models)
            
            remotecall_fetch(isfile,1,m_record.path) ? m = (remotecall_fetch(deserialize,1,m_record.path)) : (break)

            model_blacklist=copy(m.permute_blacklist)
            filteridxs, filtered_funcs, filtered_weights, filtered_args = filter_permutes(instruction, model_blacklist)

            for call in 1:instruction.func_limit
                start=time()
                funcidx=rand(filtered_weights)
                permute_func=filtered_funcs[funcidx]
                pos_args,kw_args=get_permfunc_args(permute_func,e,m,filtered_args[funcidx])
                new_m=permute_func(pos_args...;kw_args...)
                push!(call_report,(filteridxs[funcidx],time()-start,new_m.log_Li - e.contour))

                if length(new_m.permute_blacklist) > 0 && new_m.log_Li == -Inf #in this case the blacklisted function will not work on this model at all; it should be removed from the functions to use
                    vcat(model_blacklist,new_m.permute_blacklist)
                    filteridxs, filtered_funcs, filtered_weights, filtered_args = filter_permutes(instruction, model_blacklist)
                end

                dupecheck(new_m,m) && new_m.log_Li > e.contour && ((put!(models_chan, (new_m ,id, call_report))); found=true; break)
                
			end
            found==true && break;
            wait(job_chan)
            fetch(job_chan)!=e.models && (break) #if the ensemble has changed during the search, update it
			model==instruction.model_limit && (put!(models_chan, ("quit", id, call_report));persist=false)#worker to put "quit" on channel if it fails to find a model more likely than contour
		end
	end
end
                function filter_permutes(instruction, model_blacklist)
                    filteridxs=findall(p->!in(p,model_blacklist),instruction.funcs)
                    filtered_funcs=instruction.funcs[filteridxs]
                    filtered_weights=filter_weights(instruction.weights, filteridxs)
                    filtered_args=instruction.args[filteridxs]
                    return filteridxs, filtered_funcs, filtered_weights, filtered_args
                end

                function filter_weights(weights, idxs)
                    deplete=0.
                    for (i, weight) in enumerate(weights)
                        !in(i, idxs) && (deplete+=weight)
                    end
                    weights[idxs].+=deplete/length(idxs)
                    new_weights=weights[idxs]./sum(weights[idxs])
                    return Categorical(new_weights)
                end

                function get_permfunc_args(func::Function,e::IPM_Ensemble, m::ICA_PWM_Model, args::Vector{Tuple{Symbol,Any}})
                    pos_args=[]
                    argparts=Base.arg_decl_parts(methods(func).ms[1])
                    argnames=[Symbol(argparts[2][n][1]) for n in 2:length(argparts[2])]
                    for argname in argnames #assemble basic positional arguments from ensemble and model fields
                        if argname == Symbol('m')
                            push!(pos_args,m)
                        elseif argname in fieldnames(IPM_Ensemble)
                            push!(pos_args,getfield(e,argname))
                        elseif argname in fieldnames(ICA_PWM_Model)
                            push!(pos_args,getfield(m,argname))
                        else
                            throw(ArgumentError("Positional argument $argname of $func not available in the ensemble or model!"))
                        end
                    end
            
                    kw_args = NamedTuple()
                    if length(args) > 0 #if there are any keyword arguments to pass
                        sym=Vector{Symbol}()
                        val=Vector{Any}()
                        for arg in args
                            push!(sym, arg[1]); push!(val, arg[2])
                        end
                        kw_args = (;zip(sym,val)...)
                    end

                    return pos_args, kw_args
                end

                function dupecheck(new_model, model)
                    (new_model.sources==model.sources && new_model.mix_matrix==model.mix_matrix) ? (return false) : (return true)
                end