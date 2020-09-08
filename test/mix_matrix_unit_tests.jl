@testset "Mix matrix initialisation and manipulation functions" begin
    #test mix matrix init
    prior_mix_test=init_mix_matrix((trues(2,10),0.0),2, 20)
    @test all(prior_mix_test[:,1:10])
    @test !any(prior_mix_test[:,11:20])

    @test sum(init_mix_matrix((falses(0,0),1.0), O, S)) == O*S
    @test sum(init_mix_matrix((falses(0,0),0.0), O, S)) == 0
    @test 0 < sum(init_mix_matrix((falses(0,0),0.5), O, S)) < O*S

    #test mix matrix decorrelation
    empty_mixvec=falses(O)
    one_mix=mixvec_decorrelate(empty_mixvec,1)
    @test sum(one_mix)==1

    empty_mix = falses(O,S)
    new_mix,clean=mix_matrix_decorrelate(empty_mix, 500)
    @test 0 < sum(new_mix) <= 500
    @test !all(clean)

    full_mix = trues(O,S)
    less_full_mix,clean=mix_matrix_decorrelate(full_mix, 500)

    @test O*S-sum(less_full_mix) <= 500
    @test !all(clean)

    #test matrix similarity and dissimilarity functions
    test_mix=falses(O,S)
    test_mix[1:Int(floor(O/2)),:].=true
    test_idx=3
    compare_mix=deepcopy(test_mix)
    compare_mix[:,test_idx] .= .!compare_mix[:,test_idx]
    @test most_dissimilar(test_mix,compare_mix)==test_idx

    src_mixvec=falses(O)
    src_mixvec[Int(ceil(O/2)):end].=true
    @test most_similar(src_mixvec,compare_mix)==test_idx
end