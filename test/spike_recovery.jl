#Testbed for synthetic spike recovery from example background
using BioBackgroundModels, BioMotifInference, Random, Distributed, Distributions, Serialization
Random.seed!(786)
#CONSTANTS
no_obs=500
obsl=100:200

const folder_path="/bench/PhD/NGS_binaries/BBM/refined_folders"

report_folders=deserialize(folder_path)

bhmm_vec=[BHMM([0.40027008799648645, 0.1997894691941005, 0.006556173444055311, 0.3381222117369811, 0.053302496489949405, 0.0019595611384158307], [0.9910385961237116 0.001055084535731191 0.0030117366072612726 0.0025443692116936516 1.875366316590218e-83 0.002350213521605879; 0.003592562092379169 0.6212087435679351 1.4862557155404555e-91 0.3738436265294334 0.001355067810251799 5.674610926019336e-174; 0.03314627345971247 0.00026886477654034587 0.9628685354878737 3.677787936178394e-5 0.00367954839651186 5.0e-324; 0.0002903769304330447 0.3737391779744011 1.584645416437449e-35 0.6225645652084915 0.002839228767872827 0.0005666511188010834; 0.0005590565306708014 0.0064195824547637215 0.004061509889975232 0.01962586251822172 0.9687912312979926 0.0005427573083774576; 0.0174697691356862 2.0648278441425858e-20 8.008498087945918e-218 0.004151185943888868 0.0013251598095032858 0.9770538851109234], [Categorical([0.2565257692018922, 0.24506145230281004, 0.2649485969649606, 0.23346418153033835]), Categorical([0.052491168994042534, 0.0987036876775239, 0.2383241923003866, 0.610480951028046]), Categorical([0.08371833024203595, 0.07204591736665274, 0.3854013598343639, 0.4588343925569475]), Categorical([0.5954709485303519, 0.2497223085607694, 0.09528006039357192, 0.0595266825153071]), Categorical([0.4744103622376032, 0.04038264172760258, 0.04588736323302444, 0.4393196328017719]), Categorical([0.41173384083242826, 0.44270375576246973, 0.030612125665857307, 0.11495027773924743])])]

bhmm_dist=Categorical([1.])

struc_sig_1=[.1 .7 .1 .1
           .1 .1 .1 .7
           .1 .7 .1 .1]
periodicity=8
struc_frac_obs=.75


tata_box=[.05 .05 .05 .85
          .85 .05 .05 .05
          .05 .05 .05 .85
          .85 .05 .05 .05
          .425 .075 .075 .425
          .85 .05 .05 .05
          .425 .075 .075 .425]
motif_frac_obs=.7
motif_recur_range=1:4

@info "Constructing synthetic sample set..."
obs1, bg_scores1, hmm_truth1, spike_truth1 = synthetic_sample(no_obs,obsl,bhmm_vec,bhmm_dist,[struc_sig_1,tata_box],[(true,(struc_frac_obs,periodicity)),(false,(motif_frac_obs,motif_recur_range))])

e1 = "BMISpikeTest"

#JOB CONSTANTS
const ensemble_size = 750
const no_sources = 4
const source_min_bases = 3
const source_max_bases = 12
const source_length_range= source_min_bases:source_max_bases
const mixing_prior = .5
const models_to_permute = ensemble_size * 3
funcvec=full_perm_funcvec
push!(funcvec, BioMotifInference.permute_source)
push!(funcvec, BioMotifInference.permute_source)
args=[Vector{Tuple{Symbol,Any}}() for i in 1:length(funcvec)]
args[end-1]=[(:weight_shift_freq,0.),(:length_change_freq,1.),(:length_perm_range,1:1)]
args[end]=[(:weight_shift_freq,.1),(:length_change_freq,0.),(:weight_shift_dist,Uniform(.00001,.01))]


instruct = Permute_Instruct(funcvec, ones(length(funcvec))./length(funcvec),models_to_permute,100; args=args)

@info "Assembling source priors..."
prior_array= Vector{Matrix{Float64}}()
source_priors = BioMotifInference.assemble_source_priors(no_sources, prior_array)

@info "Assembling ensemble..."
isfile(e1*"/ens") ? (ens1 = deserialize(e1*"/ens")) :
    (ens1 = IPM_Ensemble(e1, ensemble_size, source_priors, (falses(0,0), mixing_prior), bg_scores1, obs1, source_length_range))
    
@info "Converging ensemble..."
logZ1 = BioMotifInference.converge_ensemble!(ens1, instruct, backup=(true,250), wk_disp=false, tuning_disp=false, ens_disp=true, conv_plot=false, src_disp=true, lh_disp=false, liwi_disp=false)
