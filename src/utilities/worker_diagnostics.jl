struct Worker_Monitor
    idx::Dict{Integer,Integer}
    persist::BitMatrix
    last_seen::Matrix{Float64}
end

function Worker_Monitor(wk_pool::Vector{<:Integer})
    idx=Dict{Integer,Integer}()
    for (n,wk) in enumerate(wk_pool)
        idx[wk]=n
    end
    persist=trues(1,length(wk_pool))
    last_seen=[time() for x in 1:1, y in 1:length(wk_pool)]
    return Worker_Monitor(idx, persist, last_seen)
end

function Base.show(io::IO, mon::Worker_Monitor; progress=false)
    printstyled("Worker Diagnostics\n", bold=true)
    pers=heatmap(float.(mon.persist), colormap=persistcolor, title="Peristence",labels=false)
    ls=heatmap([time()-ls for ls in mon.last_seen], title="Last Seen",labels=false)
    show(pers)
    println()
    show(ls)
    progress && return nrows(pers.graphics)+nrows(ls.graphics)+7
end

function persistcolor(z, zmin, zmax)
    z==1. && return 154
    z==0. && return 160
end

function update_worker_monitor!(mon,wk,persist)
    mon.persist[1,mon.idx[wk]]=persist
    mon.last_seen[1,mon.idx[wk]]=time()
end