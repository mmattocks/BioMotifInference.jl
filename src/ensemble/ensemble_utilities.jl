function ensemble_history(e::IPM_Ensemble, bins=25)
    !e.sample_posterior && throw(ArgumentError("This ensemble has no posterior samples to show a history for!"))
    livec=vcat([model.log_Li for model in e.models],[model.log_Li for model in e.retained_posterior_samples])
    show(histogram(livec, nbins=bins))
end

function e_backup(e::IPM_Ensemble, tuner::Permute_Tuner)
    serialize(string(e.path,'/',"ens"), e)
    serialize(string(e.path,'/',"tuner"), tuner)
end

function clean_ensemble_dir(e::IPM_Ensemble, model_pad::Integer; ignore_warn=false)
    !ignore_warn && e.sample_posterior && throw(ArgumentError("Ensemble is set to retain posterior samples and its directory should not be cleaned!"))
    for file in readdir(e.path)
        !(file in vcat([basename(model.path) for model in e.models],"ens",[string(number) for number in e.model_counter-length(e.models)-model_pad:e.model_counter-1])) && rm(e.path*'/'*file)
    end
end

function complete_evidence(e::IPM_Ensemble)
    return final_logZ = logaddexp(e.log_Zi[end], (logsumexp([model.log_Li for model in e.models]) +  e.log_Xi[length(e.log_Li)] - log(length(e.models))))
end

function reset_ensemble!(e::IPM_Ensemble)
    new_e=deepcopy(e)
    for i in 1:length(e.models)
        if string(i) in [basename(record.path) for record in e.models]
            new_e.models[i]=e.models[findfirst(isequal(string(i)), [basename(record.path) for record in e.models])]
        else
            new_e.models[i]=e.retained_posterior_samples[findfirst(isequal(string(i)), [basename(record.path) for record in e.retained_posterior_samples])]
        end
    end

    new_e.contour=minimum([record.log_Li for record in new_e.models])

    new_e.log_Li=[new_e.log_Li[1]]
    new_e.log_Xi=[new_e.log_Xi[1]]
    new_e.log_wi=[new_e.log_wi[1]]
    new_e.log_Liwi=[new_e.log_Liwi[1]]
    new_e.log_Zi=[new_e.log_Zi[1]]
    new_e.Hi=[new_e.Hi[1]]

    new_e.retained_posterior_samples=Vector{Model_Record}()

    new_e.model_counter=length(new_e.models)+1

    clean_ensemble_dir(new_e, 0; ignore_warn=true)
    isfile(e.path*"/tuner") && rm(e.path*"/tuner")
    serialize(e.path*"/ens", new_e)

    return new_e
end

function move_ensemble!(e::IPM_Ensemble,path::String)
    !isdir(path) && mkdir(path)
    for file in readdir(e.path)
        mv(e.path*'/'*file,path*'/'*file)
    end

    for (n,model) in enumerate(e.models)
        e.models[n]=Model_Record(path*'/'*basename(model.path), model.log_Li)
    end
    if e.sample_posterior
        for (n,model) in enumerate(e.retained_posterior_samples)
            e.retained_posterior_samples[n]=Model_Record(path*'/'*basename(model.path), model.log_Li)
        end
    end

    rm(e.path)
    e.path=path
    serialize(e.path*"/ens",e)
    return e
end

function copy_ensemble!(e::IPM_Ensemble,path::String)
    new_e=deepcopy(e)
    !isdir(path) && mkdir(path)
    for file in readdir(e.path)
        cp(e.path*'/'*file,path*'/'*file, force=true)
    end

    for (n,model) in enumerate(e.models)
        new_e.models[n]=Model_Record(path*'/'*basename(model.path), model.log_Li)
    end
    if e.sample_posterior
        for (n,model) in enumerate(e.retained_posterior_samples)
            new_e.retained_posterior_samples[n]=Model_Record(path*'/'*basename(model.path), model.log_Li)
        end
    end

    new_e.path=path
    serialize(new_e.path*"/ens",e)
    return new_e
end

function rewind_ensemble(e::IPM_Ensemble,rewind_idx)
    !e.sample_posterior && throw(ArgumentError("An ensemble not retaining posterior samples cannot be rewound!"))
    rewind_idx >= length(e.log_Li) && throw(ArgumentError("rewind_idx must be less than the current iterate!"))

    n=length(e.models)
    max_model_no=length(e.log_Li)+length(e.models)-1
    rewind_model_no=rewind_idx+length(e.models)-1
    new_e = deepcopy(e)

    rm_models=[string(name) for name in rewind_model_no+1:max_model_no]

    filter!(model->!(basename(model.path) in rm_models),new_e.models)
    filter!(model->!(basename(model.path) in rm_models),new_e.retained_posterior_samples)

    while length(new_e.models) < n
        push!(new_e.models,pop!(new_e.retained_posterior_samples))
    end
    new_e.contour=new_e.log_Li[rewind_idx]
    new_e.log_Li=new_e.log_Li[1:rewind_idx]
    new_e.log_Xi=new_e.log_Xi[1:rewind_idx]
    new_e.log_wi=new_e.log_wi[1:rewind_idx]
    new_e.log_Liwi=new_e.log_Liwi[1:rewind_idx]
    new_e.log_Zi=new_e.log_Zi[1:rewind_idx]
    new_e.Hi=new_e.Hi[1:rewind_idx]

    new_e.model_counter=length(new_e.models)+rewind_idx

    return new_e
end

function show_models(e::IPM_Ensemble,idxs)
    liperm=sortperm([model.log_Li for model in e.models],rev=true)
    for idx in idxs
        m=deserialize(e.models[liperm[idx]].path)
        show(m)
    end
end

function get_model(e::IPM_Ensemble,no)
    return deserialize(e.path*'/'*string(no))
end