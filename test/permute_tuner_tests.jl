@testset "Permute Tuner" begin
    funcvec=[random_decorrelate,fit_mix]
    instruct=Permute_Instruct(funcvec, [.5,.5],100,100)
    tuner=Permute_Tuner(instruct)
    #need to test update_weights functionality
    #want to supply some fake data to induce a .8 .2 categorical
    tuner.successes[:,1]=falses(TUNING_MEMORY*instruct.func_limit)
    tuner.successes[:,2]=falses(TUNING_MEMORY*instruct.func_limit)
    tuner.successes[1:Int(floor(TUNING_MEMORY*instruct.func_limit*.8)),1].=true
    tuner.successes[1:Int(floor(TUNING_MEMORY*instruct.func_limit*.2)),2].=true
    update_weights!(tuner)
    @test tuner.inst.weights==[.8,.2]    #check clamping
    tuner.successes[:,1]=falses(TUNING_MEMORY*instruct.func_limit)
    @test tuner.inst.min_clmps==[.01,.01]
    @test tuner.inst.max_clmps==[1.,1.]
    update_weights!(tuner)
    @test tuner.inst.weights==[.01,1-.01]

    #test override
    instruct=Permute_Instruct(funcvec, [.5,.5],100,100,override_time=2.,override_weights=[.75,.25])
    tuner=Permute_Tuner(instruct)
    tune_weights!(tuner, [(1,1.,1.)])
    @test tuner.inst.weights==[.5,.5] #no override
    tuner.time_history=fill(3.,CONVERGENCE_MEMORY)
    tune_weights!(tuner, [(1,1.,1.)])
    @test tuner.inst.weights==[.75,.25]

    #more clamping tests
    testvec=[0.02693244970132842, 0.031512823560878485, 0.050111382705776294, 0.6893314065796903, 0.04206537140157707, 0.02013161919125878, 0.026535815205008566, 0.051758654139053166, 0.03497967993105292, 0.013999262917669668, 0.01264153466670629]
    target=[0.02527900179828276, 0.029859375657832827, 0.04845793480273063, 0.6876779586766446, 0.040411923498531406, 0.02, 0.02488236730196291, 0.050105206236007505, 0.03332623202800726, 0.02, 0.02]

    clamp_pvec!(testvec,fill(.02,length(testvec)),fill(1.,length(testvec)))
    @test isprobvec(testvec)
    @test testvec==target

    #high clamp
    testvec=[.9,.08,.02]
    clamp_pvec!(testvec, fill(.05,length(testvec)), [.45,.45,.45])
    @test testvec==[.45, .2825, .2675]
end
