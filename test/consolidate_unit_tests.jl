@testset "Orthogonality helper" begin
    bg_scores = log.(fill(.25, (17,3)))
    obs=[BioSequences.LongSequence{DNAAlphabet{2}}("ATGATTACGATGATGCA")
    BioSequences.LongSequence{DNAAlphabet{2}}("TCAGTTACGATGATCAG")
    BioSequences.LongSequence{DNAAlphabet{2}}("TTACGCACAGATGTTAC")]
    order_seqs = BioBackgroundModels.get_order_n_seqs(obs, 0)
    coded_seqs = BioBackgroundModels.code_seqs(order_seqs)
    obs=Array(transpose(coded_seqs))
    obsl=[findfirst(iszero,obs[:,o])-1 for o in 1:size(obs)[2]]

    src_ATG = [.7 .1 .1 .1
    .1 .1 .1 .7
    .1 .1 .7 .1]

    src_CAG = [.1 .7 .1 .1
    .7 .1 .1 .1
    .1 .1 .7 .1]

    src_TTAC = [.1 .1 .1 .7
    .1 .1 .1 .7
    .7 .1 .1 .1
    .1 .7 .1 .1]

    src_GCA = [.1 .1 .7 .1
    .1 .7 .1 .1
    .7 .1 .1 .1]

    consolidate_one = [(log.(src_ATG),0),(log.(src_TTAC),0),(log.(src_ATG),0)]
    cons_one_mix = BitMatrix([true true false
                    true false true
                    false true true])

    consolidate_two = [(log.(src_ATG),0),(log.(src_ATG),0),(log.(src_ATG),0)]
    cons_two_mix = BitMatrix([true false false
                    false true false
                    false false true])

    distance_model = [(log.(src_TTAC),0),(log.(src_CAG),0),(log.(src_GCA),0)]

    c1model = ICA_PWM_Model("c1", "c1", consolidate_one, 3:4, cons_one_mix, IPM_likelihood(consolidate_one, obs, obsl, bg_scores, cons_one_mix))
    c2model = ICA_PWM_Model("c2", "c2", consolidate_two, 3:4, cons_two_mix, IPM_likelihood(consolidate_two, obs, obsl, bg_scores, cons_two_mix))
    dmodel = ICA_PWM_Model("d", "d", distance_model, 3:4, trues(3,3), IPM_likelihood(distance_model, obs, obsl, bg_scores, trues(3,3)))

    dpath=randstring()
    serialize(dpath, dmodel)
    drec=Model_Record(dpath,dmodel.log_Li)

    #check distance calculation
    pwmtest_1=zeros(1,4);pwmtest_1[1]=1.
    pwmtest_2=zeros(1,4);pwmtest_2[2]=1.
    @test pwm_distance(log.(pwmtest_1),log.(pwmtest_2)) == euclidean(pwmtest_1,pwmtest_2) == 1.4142135623730951

    #test consolidate check
    @test consolidate_check(distance_model) == (true,Dict{Integer,Vector{Integer}}())
    @test consolidate_check(consolidate_one) == (false, Dict{Integer,Vector{Integer}}(1=>[3]))
    @test consolidate_check(consolidate_two) == (false, Dict{Integer,Vector{Integer}}(1=>[2,3], 2=>[3]))

    #test overall consolidate function
    _,con_idxs=consolidate_check(consolidate_one)
    c1consmod=consolidate_srcs(con_idxs, c1model, obs, obsl, bg_scores, drec.log_Li, [drec])

    @test consolidate_check(c1consmod.sources)[1]
    @test c1consmod.log_Li > dmodel.log_Li
    @test all(c1consmod.mix_matrix[:,1])
    @test c1consmod.sources[1][1]==log.(src_ATG)
    @test c1consmod.sources[3][1]!=log.(src_ATG)
    @test c1consmod.origin=="consolidated c1"

    _,con_idxs=consolidate_check(consolidate_two)
    c2consmod=consolidate_srcs(con_idxs, c2model, obs, obsl, bg_scores, drec.log_Li, [drec])

    @test consolidate_check(c2consmod.sources)[1]
    @test c2consmod.log_Li > dmodel.log_Li
    @test all(c2consmod.mix_matrix[:,1])
    @test c2consmod.sources[1][1]==log.(src_ATG)
    @test c2consmod.sources[2][1]!=log.(src_ATG)
    @test c2consmod.sources[3][1]!=log.(src_ATG)
    @test c2consmod.origin=="consolidated c2"

    rm(dpath)
end
