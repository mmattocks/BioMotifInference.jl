function sequence_workers(wk_pool, e, model_chan, job_chan)
    comms_chan = RemoteChannel(()->Channel{Integer}(length(wk_pool)))

    for worker in wk_pool
        remote_do(permute_IPM, worker, e, job_chan, model_chan, comms_chan)
        wait(comms_chan)
        report=take!(comms_chan)
        @assert worker==report
    end
end