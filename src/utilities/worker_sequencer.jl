#waits for one worker to begin before calling the next so that network is not slammed trying to send huge arrays to many workers simultaneously

function sequence_workers(wk_pool, func, args...)
    comms_chan = RemoteChannel(()->Channel{Integer}(length(wk_pool)))

    for worker in wk_pool
        remote_do(func, worker, args..., comms_chan)
        wait(comms_chan); report=take!(comms_chan)
        @assert worker==report
    end
end