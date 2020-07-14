function converge_ensemble!(e::IPM_Ensemble, instruction::Permute_Instruct, evidence_fraction::AbstractFloat=.001; min_func_weight=.01, backup::Tuple{Bool,Integer}=(false,0), verbose::Bool=false, progargs...)
    N = length(e.models)
    log_frac=log(evidence_fraction)
    
    tuner = Permute_Tuner(instruction,min_func_weight);
    wk_mon = Worker_Monitor([1]);
    meter = ProgressNS(e, wk_mon, tuner, 0., log_frac; start_it=length(e.log_Li), progargs...)


    while lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])
        iterate = length(e.log_Li) #get the iterate from the enemble 
        warn,step_report = nested_step!(e, instruction) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "Failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info
        iterate += 1

        tune_weights!(tuner, step_report)
        instruction = tune_instruction(tuner, instruction)

        backup[1] && iterate%backup[2] == 0 && serialize(string(e.path,'/',"ens"), e) #every backup interval, serialise the ensemble

        update!(meter,lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]),lps(log_frac,e.log_Zi[end]))        
    end

    final_logZ = logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end

    
function converge_ensemble!(e::IPM_Ensemble, instruction::Permute_Instruct, clerk::Vector{Int64}, wk_pool::Vector{Int64}, evidence_fraction::AbstractFloat=.001; min_func_weight=.01, backup::Tuple{Bool,<:Integer}=(false,0), verbose::Bool=false, progargs...)
    length(clerk)>1 && @warn "Only one clerk worker process is used!"
    clerk=clerk[1]

    N = length(e.models)
    log_frac=log(evidence_fraction)
    
    model_chan= RemoteChannel(()->Channel{Tuple{Union{ICA_PWM_Model,Nothing},Integer, AbstractVector{<:Tuple}}}(length(wk_pool))) #channel to take EM iterates off of
    job_chan = RemoteChannel(()->Channel{Tuple{<:AbstractVector{<:Model_Record}, Float64, Union{Permute_Instruct,Nothing}}}(1))
    put!(job_chan,(e.models, e.contour, instruction))

    for (x,worker) in enumerate(wk_pool)
        remote_do(permute_IPM, worker, e, job_chan, model_chan)
    end
    
    wk_mon=Worker_Monitor(wk_pool)
    tuner = Permute_Tuner(instruction,min_func_weight)
    meter = ProgressNS(e, wk_mon, tuner, 0., log_frac; start_it=length(e.log_Li), progargs...)

    while lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])
        iterate = length(e.log_Li) #get the iterate from the ensemble
        #REMOVE OLD LEAST LIKELY MODEL - perform here to spare all workers the same calculations
        e.contour, least_likely_idx = findmin([model.log_Li for model in e.models])
        Li_model = e.models[least_likely_idx]
        deleteat!(e.models, least_likely_idx)
        e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

        warn, step_report = nested_step!(e, model_chan, wk_mon, Li_model) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "All workers failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info
        iterate += 1
        tune_weights!(tuner, step_report)
        instruction = tune_instruction(tuner, instruction) 
        take!(job_chan); put!(job_chan,(e.models,e.contour,instruction))
        backup[1] && iterate%backup[2] == 0 && serialize(string(e.path,'/',"ens"), e) #every backup interval, serialise the ensemble
        update!(meter, lps(findmax([model.log_Li for model in e.models])[1], e.log_Xi[end]), lps(log_frac,e.log_Zi[end]))
    end
    take!(job_chan); put!(job_chan, (e.models, e.contour, nothing)) #nothing instruction terminates worker functions
    final_logZ = logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(1/length(e.models))

    @info "Job done, sampled to convergence. Final logZ $final_logZ"

    return final_logZ
end