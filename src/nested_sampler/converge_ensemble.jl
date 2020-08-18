function converge_ensemble!(e::IPM_Ensemble, instruction::Permute_Instruct, evidence_fraction::AbstractFloat=.001; max_iterates=typemax(Int64), backup::Tuple{Bool,Integer}=(false,0), verbose::Bool=false, progargs...)
    N = length(e.models)
    log_frac=log(evidence_fraction)
    
    curr_it=length(e.log_Li)
    tuner = Permute_Tuner(instruction);
    wk_mon = Worker_Monitor([1]);
    meter = ProgressNS(e, wk_mon, tuner, 0., log_frac; start_it=curr_it, progargs...)


    while (lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])) && (curr_it <= max_iterates)
        warn,step_report = nested_step!(e, instruction) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "Failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info
        curr_it += 1

        tune_weights!(tuner, step_report)
        instruction = tune_instruction(tuner, instruction)

        backup[1] && curr_it%backup[2] == 0 && serialize(string(e.path,'/',"ens"), e) #every backup interval, serialise the ensemble

        update!(meter,lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]),lps(log_frac,e.log_Zi[end]))        
    end

    if (lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) < lps(log_frac,e.log_Zi[end]))
        final_logZ = logaddexp(e.log_Zi[end], (logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(length(e.models))))
        @info "Job done, sampled to convergence. Final logZ $final_logZ"

        serialize(string(e.path,'/',"ens"), e)
        return final_logZ
    elseif curr_it==max_iterates
        @info "Job done, sampled to maximum iterate $max_iterates. Convergence criterion not obtained."

        serialize(string(e.path,'/',"ens"), e)
        return e.log_Zi[end]
    end
end

    
function converge_ensemble!(e::IPM_Ensemble, instruction::Permute_Instruct, wk_pool::Vector{Int64}, evidence_fraction::AbstractFloat=.001; max_iterates=typemax(Int64), backup::Tuple{Bool,<:Integer}=(false,0), verbose::Bool=false, progargs...)
    N = length(e.models)
    log_frac=log(evidence_fraction)
    
    model_chan= RemoteChannel(()->Channel{Tuple{Union{ICA_PWM_Model,Nothing},Integer, AbstractVector{<:Tuple}}}(10*length(wk_pool))) #channel to take EM iterates off of
    job_chan = RemoteChannel(()->Channel{Tuple{<:AbstractVector{<:Model_Record}, Float64, Union{Permute_Instruct,Nothing}}}(1))
    put!(job_chan,(e.models, e.contour, instruction))

    @async sequence_workers(wk_pool, permute_IPM, e, job_chan, model_chan)
    
    curr_it=length(e.log_Li)

    wk_mon=Worker_Monitor(wk_pool)
    tuner = Permute_Tuner(instruction)
    meter = ProgressNS(e, wk_mon, tuner, 0., log_frac; start_it=curr_it, progargs...)

    #while lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])
    while (lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) >= lps(log_frac,e.log_Zi[end])) && (curr_it <= max_iterates)

        #REMOVE OLD LEAST LIKELY MODEL - perform here to spare all workers the same calculations
        e.contour, least_likely_idx = findmin([model.log_Li for model in e.models])
        Li_model = e.models[least_likely_idx]
        deleteat!(e.models, least_likely_idx)
        e.sample_posterior ? push!(e.retained_posterior_samples, Li_model) : rm(Li_model.path) #if sampling posterior, push the model record to the ensemble's posterior samples vector, otherwise delete the serialised model pointed to by the model record

        warn, step_report = nested_step!(e, model_chan, wk_mon, Li_model) #step the ensemble
        warn == 1 && #"1" passed for warn code means no workers persist; all have hit the permute limit
                (@error "All workers failed to find new models, aborting at current iterate."; return e) #if there is a warning, iust return the ensemble and print info
        curr_it += 1
        tune_weights!(tuner, step_report)
        instruction = tune_instruction(tuner, instruction) 
        take!(job_chan); put!(job_chan,(e.models,e.contour,instruction))
        backup[1] && curr_it%backup[2] == 0 && serialize(string(e.path,'/',"ens"), e) #every backup interval, serialise the ensemble
    
        update!(meter, lps(findmax([model.log_Li for model in e.models])[1], e.log_Xi[end]), lps(log_frac,e.log_Zi[end]))
    end

    take!(job_chan); put!(job_chan, (e.models, e.contour, nothing)) #nothing instruction terminates worker functions

    if (lps(findmax([model.log_Li for model in e.models])[1],  e.log_Xi[end]) < lps(log_frac,e.log_Zi[end]))
        final_logZ = logaddexp(e.log_Zi[end], (logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(length(e.models))))
        @info "Job done, sampled to convergence. Final logZ $final_logZ"

        serialize(string(e.path,'/',"ens"), e)
        return final_logZ
    elseif curr_it==max_iterates
        @info "Job done, sampled to maximum iterate $max_iterates. Convergence criterion not obtained."

        serialize(string(e.path,'/',"ens"), e)
        return e.log_Zi[end]
    end
end