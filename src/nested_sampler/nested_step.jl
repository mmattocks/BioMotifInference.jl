#### IMPLEMENTATION OF JOHN SKILLINGS' NESTED SAMPLING ALGORITHM ####
function nested_step!(e::IPM_Ensemble, instruction::Permute_Instruct)
    N = length(e.models) #number of sample models/particles on the posterior surface
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    e.contour, least_likely_idx = findmin([model.log_Li for model in e.models])
    Li_model = e.models[least_likely_idx]

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND PUSH TO ENSEMBLE
    model_selected=false; step_report=0
    while !model_selected
        candidate,step_report=permute_IPM(e, instruction)
        if !(candidate===nothing)
            if !(candidate.log_Li in [m.log_Li for m in e.models])
                model_selected=true

                new_model_record = Model_Record(string(e.path,'/',e.model_counter), candidate.log_Li);
                e.sample_posterior && push!(e.posterior_samples, Li_model)#if sampling posterior, push the model record to the ensemble's posterior samples vector
                deleteat!(e.models, least_likely_idx)
                push!(e.models, new_model_record);

                final_model=ICA_PWM_Model(string(e.model_counter), candidate.origin, candidate.sources, candidate.source_length_limits, candidate.mix_matrix, candidate.log_Li, candidate.permute_blacklist)
                serialize(new_model_record.path, final_model)

                e.model_counter +=1
            end
        else
            push!(e.models, Li_model)
            return 1, step_report
        end
    end
         
    #UPDATE ENSEMBLE QUANTITIES   
    push!(e.log_Li, e.contour) #log likelihood of the least likely model - the current ensemble ll contour at Xi
    push!(e.log_Xi, -i/N) #log Xi - crude estimate of the iterate's enclosed prior mass
    push!(e.log_wi, logsubexp(e.log_Xi[i], -j/N′) - log(2)) #log width of prior mass spanned by the last step-trapezoidal approx
    push!(e.log_Liwi, lps(e.log_Li[j],e.log_wi[j])) #log likelihood + log width = increment of evidence spanned by iterate
    push!(e.log_Zi, logaddexp(e.log_Zi[i],e.log_Liwi[j]))    #log evidence
    #information- dimensionless quantity. cf. MultiNest @ https://github.com/farhanferoz/MultiNest/blob/master/MultiNest_v3.12/nested.F90- MultiNest now gives info only at the final step
    Hj=lps( 
        (exp(lps(e.log_Liwi[j],-e.log_Zi[j])) * e.log_Li[j]), #information contribution of this step
        (exp(lps(e.log_Zi[i],-e.log_Zi[j])) * lps(e.Hi[i],e.log_Zi[i])), #rescale last information by evidence
        -e.log_Zi[j])
    Hj === -Inf ? push!(e.Hi,0.) : push!(e.Hi, Hj) #prevent problems from early strings of models with -Inf log likelihoods

    return 0, step_report
end

function nested_step!(e::IPM_Ensemble, model_chan::RemoteChannel, wk_mon::Worker_Monitor)
    N = length(e.models)+1 #number of sample models/particles on the posterior surface- +1 as one has been removed in the distributed dispatch for converge_ensemble
    i = length(e.log_Li) #iterate number, index for last values
    j = i+1 #index for newly pushed values

    e.contour, least_likely_idx = findmin([model.log_Li for model in e.models])
    Li_model = e.models[least_likely_idx]

    #SELECT NEW MODEL, SAVE TO ENSEMBLE DIRECTORY, CREATE RECORD AND REPLACE LEAST LIKELY MODEL
    model_selected=false;wk=0;step_report=0
    while !model_selected
        @async wait(model_chan)
        candidate,wk,step_report = take!(model_chan)
        if !(candidate=="quit")
            if (candidate.log_Li > e.contour) && !(candidate.log_Li in [m.log_Li for m in e.models])
                model_selected=true

                new_model_record = Model_Record(string(e.path,'/',e.model_counter), candidate.log_Li);
                e.sample_posterior && push!(e.posterior_samples, Li_model)#if sampling posterior, push the model record to the ensemble's posterior samples vector
                deleteat!(e.models, least_likely_idx)
                push!(e.models, new_model_record);

                final_model=ICA_PWM_Model(string(e.model_counter), candidate.origin, candidate.sources, candidate.source_length_limits, candidate.mix_matrix, candidate.log_Li, candidate.permute_blacklist)
                serialize(new_model_record.path, final_model)

                e.model_counter +=1
            end
            update_worker_monitor!(wk_mon,wk,true)
        else
            update_worker_monitor!(wk_mon,wk,false)
            !any(wk_mon.persist) && (return 1,step_report)
        end
    end
    
    #UPDATE ENSEMBLE QUANTITIES   
    push!(e.log_Li, minimum([model.log_Li for model in e.models])) #log likelihood of the least likely model - the current ensemble ll contour at Xi
    push!(e.log_Xi, -i/N) #log Xi - crude estimate of the iterate's enclosed prior mass
    push!(e.log_wi, lps(e.log_Xi[i], -((j+1)/N)-log(2))) #log width of prior mass spanned by the last step-trapezoidal approx
    push!(e.log_Liwi, lps(e.log_Li[j],e.log_wi[j])) #log likelihood + log width = increment of evidence spanned by iterate
    push!(e.log_Zi, logaddexp(e.log_Zi[i],e.log_Liwi[j]))    #log evidence

    #information- dimensionless quantity
    push!(e.Hi, lps(
            (exp(lps(e.log_Liwi[j],-e.log_Zi[j])) * e.log_Li[j]), #term1
            (exp(lps(e.log_Zi[i],-e.log_Zi[j])) * lps(e.Hi[i],e.log_Zi[i])), #term2
            -e.log_Zi[j])) #term3

    return 0, step_report
end