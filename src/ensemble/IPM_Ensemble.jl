mutable struct IPM_Ensemble
	path::String #ensemble models and popped-out posterior samples serialised here
	models::Vector{Model_Record} #ensemble keeps paths to serialised models and their likelihood tuples rather than keeping the models in memory

	contour::AbstractFloat#current log_Li[end]
	log_Li::Vector{AbstractFloat} #likelihood of lowest-ranked model at iterate i
	log_Xi::Vector{AbstractFloat} #amt of prior mass included in ensemble contour at Li
	log_wi::Vector{AbstractFloat} #width of prior mass covered in this iterate
	log_Liwi::Vector{AbstractFloat} #evidentiary weight of the iterate
	log_Zi::Vector{AbstractFloat} #ensemble evidence
	Hi::Vector{AbstractFloat} #ensemble information

	obs_array::Matrix{Integer} #observations Txo
	obs_lengths::Vector{Integer}

	source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}} #source pwm priors
	mix_prior::Tuple{BitMatrix,AbstractFloat} #prior on %age of observations that any given source contributes to

	bg_scores::Matrix{AbstractFloat} #precalculated background HMM scores, same dims as obs

	sample_posterior::Bool
	retained_posterior_samples::Vector{Model_Record} #list of posterior sample records

	model_counter::Integer

	naive_lh::AbstractFloat #the likelihood of the background model without any sources
end

####IPM_Ensemble FUNCTIONS
IPM_Ensemble(path::String, no_models::Integer, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractMatrix{<:AbstractFloat}, obs::AbstractArray{<:Integer}, source_length_limits; posterior_switch::Bool=false) =
IPM_Ensemble(
	path,
	assemble_IPMs(path, no_models, source_priors, mix_prior, bg_scores, obs, source_length_limits)...,
	[-Inf], #L0 = 0
	[0], #ie exp(0) = all of the prior is covered
	[-Inf], #w0 = 0
	[-Inf], #Liwi0 = 0
	[-1e300], #Z0 = 0
	[0], #H0 = 0,
	obs,
	[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]],
	source_priors,
	mix_prior,
	bg_scores, #precalculated background score
	posterior_switch,
	Vector{String}(),
	no_models+1,
	IPM_likelihood(init_logPWM_sources(source_priors, source_length_limits), obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores, falses(size(obs)[2],length(source_priors))))

IPM_Ensemble(worker_pool::AbstractVector{<:Integer}, path::String, no_models::Integer, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractMatrix{<:AbstractFloat}, obs::Array{<:Integer}, source_length_limits; posterior_switch::Bool=false) =
IPM_Ensemble(
	path,
	distributed_IPM_assembly(worker_pool, path, no_models, source_priors, mix_prior, bg_scores, obs, source_length_limits)...,
	[-Inf], #L0 = 0
	[0], #ie exp(0) = all of the prior is covered
	[-Inf], #w0 = 0
	[-Inf], #Liwi0 = 0
	[-1e300], #Z0 = 0
	[0], #H0 = 0,
	obs,
	[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]],
	source_priors,
	mix_prior,
	bg_scores, #precalculated background score
	posterior_switch,
	Vector{String}(),
	no_models+1,
	IPM_likelihood(init_logPWM_sources(source_priors, source_length_limits), obs, [findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]], bg_scores, falses(size(obs)[2],length(source_priors))))

function assemble_IPMs(path::String, no_models::Integer, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractArray{<:AbstractFloat}, obs::AbstractArray{<:Integer}, source_length_limits::UnitRange{<:Integer})
	ensemble_records = Vector{Model_Record}()
	!isdir(path) && mkpath(path)

	@assert size(obs)[2]==size(bg_scores)[2]

	@showprogress 1 "Assembling IPM ensemble..." for model_no in 1:no_models
		model_path = string(path,'/',model_no)
		if !isfile(model_path)
			model = ICA_PWM_Model(string(model_no), source_priors, mix_prior, bg_scores, obs, source_length_limits)
			serialize(model_path, model) #save the model to the ensemble directory
			push!(ensemble_records, Model_Record(model_path,model.log_Li))
		else #interrupted assembly pick up from where we left off
			model = deserialize(model_path)
			push!(ensemble_records, Model_Record(model_path,model.log_Li))
		end
	end

	return ensemble_records, minimum([record.log_Li for record in ensemble_records])
end

function distributed_IPM_assembly(worker_pool::Vector{Int64}, path::String, no_models::Integer, source_priors::AbstractVector{<:Union{<:AbstractVector{<:Dirichlet{<:AbstractFloat}},<:Bool}}, mix_prior::Tuple{BitMatrix,<:AbstractFloat}, bg_scores::AbstractArray{<:AbstractFloat}, obs::AbstractArray{<:Integer}, source_length_limits::UnitRange{<:Integer})
	ensemble_records = Vector{Model_Record}()
	!isdir(path) && mkpath(path)

	@assert size(obs)[2]==size(bg_scores)[2]

    model_chan= RemoteChannel(()->Channel{ICA_PWM_Model}(length(worker_pool)))
    job_chan = RemoteChannel(()->Channel{Union{Tuple,Nothing}}(1))
	put!(job_chan,(source_priors, mix_prior, bg_scores, obs, source_length_limits))
	
	sequence_workers(worker_pool, worker_assemble, job_chan, model_chan)
	
	assembly_progress=Progress(no_models, desc="Assembling IPM ensemble...")

	model_counter=check_assembly!(ensemble_records, path, no_models, assembly_progress)

	while model_counter <= no_models
		wait(model_chan)
		candidate=take!(model_chan)
		model = ICA_PWM_Model(string(model_counter), candidate.origin, candidate.sources, candidate.source_length_limits, candidate.mix_matrix, candidate.log_Li)
		model_path=string(path,'/',model_counter)
		serialize(model_path,model)
		push!(ensemble_records, Model_Record(model_path,model.log_Li))
		model_counter+=1
		next!(assembly_progress)
	end

	take!(job_chan),put!(job_chan,nothing)

	return ensemble_records, minimum([record.log_Li for record in ensemble_records])
end
				function check_assembly!(ensemble_records::AbstractVector{<:Model_Record}, path::String, no_models::Integer, assembly_progress::Progress)
					counter=1
					while counter <= no_models
						model_path=string(path,'/',counter)
						if isfile(model_path)
							model=deserialize(model_path)
							push!(ensemble_records, Model_Record(model_path,model.log_Li))
							counter+=1
							next!(assembly_progress)
						else
							return counter
						end
					end
					return counter
				end

				function worker_assemble(job_chan::RemoteChannel, models_chan::RemoteChannel, comms_chan::RemoteChannel)
					put!(comms_chan,myid())
					wait(job_chan)
					params=fetch(job_chan)
					while !(fetch(job_chan) === nothing)
						model=ICA_PWM_Model(string(myid()),params...)
						put!(models_chan,model)
					end
				end

function Base.show(io::IO, e::IPM_Ensemble; nsrc=0, progress=false)
	livec=[model.log_Li for model in e.models]
	maxLH=maximum(livec)
	printstyled(io, "ICA PWM Model Ensemble @ $(e.path)\n", bold=true)
	msg = @sprintf "Contour: %3.3e MaxLH:%3.3e Max/Naive:%3.3e log Evidence:%3.3e" e.contour maxLH (maxLH-e.naive_lh) e.log_Zi[end]
	println(io, msg)
	hist=UnicodePlots.histogram(livec, title="Ensemble Likelihood Distribution")
	show(io, hist)
	println()
	progress && return(nrows(hist.graphics)+6)
end