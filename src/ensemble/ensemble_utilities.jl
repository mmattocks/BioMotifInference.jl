function clean_ensemble_dir(e::IPM_Ensemble)
    e.sample_posterior && throw(ArgumentError("Ensemble is set to retain posterior samples and its directory should not be cleaned!"))
    for file in readdir(e.path)
        !(file in vcat([basename(model.path) for model in e.models],"ens")) && rm(e.path*'/'*file)
    end
end

function reset_ensemble(e::IPM_Ensemble)
    new_e=deepcopy(e)
    for i in 1:length(e.models)
        if string(i) in [basename(record.path) for record in e.models]
            new_e.models[i]=e.models[findfirst(isequal(string(i)), [basename(record.path) for record in e.models])]
        else
            new_e.models[i]=e.retained_posterior_samples[findfirst(isequal(string(i)), [basename(record.path) for record in e.retained_posterior_samples])]
        end
    end

    new_e.contour=new_e.log_Li[1]
    new_e.log_Li=[new_e.log_Li[1]]
    new_e.log_Xi=[new_e.log_Xi[1]]
    new_e.log_wi=[new_e.log_wi[1]]
    new_e.log_Liwi=[new_e.log_Liwi[1]]
    new_e.log_Zi=[new_e.log_Zi[1]]
    new_e.Hi=[new_e.Hi[1]]

    new_e.retained_posterior_samples=Vector{Model_Record}()

    new_e.model_counter=length(new_e.models)+1

    return new_e
end