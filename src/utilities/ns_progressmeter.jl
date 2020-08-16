#UTILITY  REPORTS WORKER NUMBER AND CURRENT ITERATE
mutable struct ProgressNS{T<:Real} <: AbstractProgress
    interval::T
    dt::AbstractFloat
    start_it::Integer
    counter::Integer
    triggered::Bool
    tfirst::AbstractFloat
    tlast::AbstractFloat
    tstp::AbstractFloat
    printed::Bool        # true if we have issued at least one status update
    desc::AbstractString # prefix to the percentage, e.g.  "Computing..."
    color::Symbol        # default to green
    output::IO           # output stream into which the progress is written
    numprintedvalues::Integer   # num values printed below progress in last iteration
    offset::Integer             # position offset of progress bar (default is 0)

    e::IPM_Ensemble
    wm::Worker_Monitor
    tuner::Permute_Tuner
    top_m::ICA_PWM_Model    

    log_frac::AbstractFloat
    mean_stp_time::AbstractFloat

    wk_disp::Bool
    tuning_disp::Bool
    conv_plot::Bool
    ens_disp::Bool
    lh_disp::Bool
    liwi_disp::Bool
    src_disp::Bool
    no_displayed_srcs::Integer

    disp_rotate_inst::Vector{Any}
    
    convergence_history::Vector{Float64}
    time_history::Vector{Float64}

    function ProgressNS{T}(    e::IPM_Ensemble,
                               top_m::ICA_PWM_Model,
                               wm::Worker_Monitor,
                               tuner::Permute_Tuner,
                               log_frac::AbstractFloat,
                               interval::T;
                               dt::Real=0.1,
                               desc::AbstractString="Nested Sampling::",
                               color::Symbol=:green,
                               output::IO=stdout,
                               offset::Int=0,
                               start_it::Int=1,
                               wk_disp::Bool=false,
                               tuning_disp::Bool=false,
                               conv_plot::Bool=true,
                               ens_disp::Bool=false,
                               lh_disp::Bool=false,
                               liwi_disp::Bool=false,
                               src_disp::Bool=true,
                               nsrcs::Integer=0,
                               disp_rotate_inst=[false,0,0,Vector{Vector{Symbol}}()]) where T
        tfirst = tlast = time()
        printed = false
        new{T}(interval,
         dt,
         start_it,
         start_it,
         false,
         tfirst,
         tlast,
         0., 
         printed, 
         desc,
         color,
         output,
         0,
         offset,
         e,
         wm,
         tuner,
         top_m,
         log_frac,
         0.,
         wk_disp,
         tuning_disp,
         conv_plot,
         ens_disp,
         lh_disp,
         liwi_disp,
         src_disp,
         nsrcs,
         disp_rotate_inst,
         zeros(CONVERGENCE_MEMORY),
         zeros(CONVERGENCE_MEMORY))
    end
end

function ProgressNS(e::IPM_Ensemble, wm::Worker_Monitor, tuner::Permute_Tuner, interval::Real, log_frac::AbstractFloat; dt::Real=0.1, desc::AbstractString="Nested Sampling::", color::Symbol=:green, output::IO=stderr, offset::Integer=0, start_it::Integer=1, wk_disp::Bool=false, tuning_disp::Bool=false, conv_plot::Bool=true, lh_disp::Bool=false, liwi_disp::Bool=false, ens_disp::Bool=false, src_disp::Bool=true, nsrcs=0, disp_rotate_inst::Vector{Any}=[false,0,0,Vector{Vector{Symbol}}()])   
    top_m = deserialize(e.models[findmax([model.log_Li for model in e.models])[2]].path)
    
    return ProgressNS{typeof(interval)}(e, top_m, wm, tuner, log_frac, interval, dt=dt, desc=desc, color=color, output=output, offset=offset, start_it=start_it, wk_disp=wk_disp, tuning_disp=tuning_disp, conv_plot=conv_plot, lh_disp=lh_disp, liwi_disp=liwi_disp, ens_disp=ens_disp, src_disp=src_disp, nsrcs=nsrcs, disp_rotate_inst=disp_rotate_inst)
end


function update!(p::ProgressNS, val, thresh; options...)
    p.counter += 1
    
    p.tstp=time()-p.tlast
    popfirst!(p.time_history)
    push!(p.time_history, p.tstp)

    p.interval = val - thresh
    popfirst!(p.convergence_history)
    push!(p.convergence_history, p.interval)    

    p.top_m.name != basename(p.e.models[findmax([model.log_Li for model in p.e.models])[2]].path) && (p.top_m=deserialize(p.e.models[findmax([model.log_Li for model in p.e.models])[2]].path))

    updateProgress!(p; options...)
end

function updateProgress!(p::ProgressNS; offset::Integer = p.offset, keep = (offset == 0))
    p.offset = offset
    t = time()
    p.disp_rotate_inst[1] && p.counter%p.disp_rotate_inst[2] == 0 && rotate_displays(p)

    if p.interval <= 0 && !p.triggered
        p.triggered = true
        if p.printed
            p.triggered = true
            dur = durationstring(t-p.tfirst)
            msg = @sprintf "%s Converged. Time: %s (%d iterations). logZ: %s\n" p.desc dur p.counter p.e.log_Zi[end]
            
            print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
            move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
            upper_lines=display_upper_dash(p)
            printover(p.output, msg, :magenta)
            lower_lines=display_lower_dash(p)
            
            p.numprintedvalues=upper_lines + lower_lines + 1

            if keep
                println(p.output)
            else
                print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))
            end
        end
        return
    end

    if t > p.tlast+p.dt && !p.triggered
        msg = @sprintf "%s Iterate: %s Recent step time μ: %s Convergence Interval: %g\n" p.desc p.counter hmss(mean(p.time_history)) p.interval

        print(p.output, "\n" ^ (p.offset + p.numprintedvalues))
        move_cursor_up_while_clearing_lines(p.output, p.numprintedvalues)
        upper_lines=display_upper_dash(p)
        printover(p.output, msg, p.color)
        lower_lines=display_lower_dash(p)

        p.numprintedvalues=upper_lines + lower_lines + 1
        print(p.output, "\r\u1b[A" ^ (p.offset + p.numprintedvalues))

        # Compensate for any overhead of printing. This can be
        # especially important if you're running over a slow network
        # connection.
        p.tlast = t + 2*(time()-t)
        p.printed = true
    end
end
                function rotate_displays(p)
                    curr_inst=p.disp_rotate_inst[3]
                    curr_inst+1>length(p.disp_rotate_inst[4]) ? next_inst=1 : next_inst=curr_inst+1
                    next_disps=p.disp_rotate_inst[4][next_inst]
                    for disp in [:wk_disp,:tuning_disp,:conv_plot,:ens_disp,:lh_disp,:liwi_disp,:src_disp]
                        disp in next_disps ?  setproperty!(p, disp, true) : setproperty!(p, disp, false)
                    end
                    p.disp_rotate_inst[3]=next_inst
                end

                function hmss(dt)
                    dt<0 ? (dt=-dt; prfx="-") : (prfx="")
                    isnan(dt) && return "NaN"
                    (h,r) = divrem(dt,60*60)
                    (m,r) = divrem(r, 60)
                    (isnan(h)||isnan(m)||isnan(r)) && return "NaN"
                    string(prfx,Int(h),":",Int(m),":",Int(ceil(r)))
                end

                function display_upper_dash(p::ProgressNS)
                    wklines = tunelines = cilines = 0
                    p.wk_disp && (wklines=show(p.output, p.wm, progress=true);println())
                    p.tuning_disp && (tunelines=show(p.output, p.tuner, progress=true);println())
                    if p.conv_plot
                        ciplot=lineplot([p.counter-(length(p.convergence_history)-1):p.counter...], p.convergence_history, title="Convergence Interval Recent History", xlabel="Iterate",ylabel="CI", color=:yellow)
                        cilines=nrows(ciplot.graphics)+5
                        show(p.output, ciplot); println()
                    end
                    return wklines + tunelines + cilines
                end

                function display_lower_dash(p::ProgressNS)
                    lhlines = liwilines = ensemblelines = srclines = 0
                    if p.lh_disp
                        lhplot=lineplot(p.e.log_Li[2:end], title="Contour History", xlabel="Iterate", color=:magenta, name="Ensemble logLH")
                        lineplot!(lhplot, [p.e.naive_lh for it in 1:length(p.e.log_Li[2:end])], name="Naive logLH")
                        lhlines=nrows(lhplot.graphics)+5
                        show(p.output, lhplot); println()
                    end
                    if p.liwi_disp && p.counter>2
                        liwiplot=lineplot([max(2,p.counter-(CONVERGENCE_MEMORY-1)):p.counter...],p.e.log_Liwi[max(2,end-(CONVERGENCE_MEMORY-1)):end], title="Recent iterate evidentiary weight", xlabel="Iterate", name="Ensemble log Liwi", color=:cyan)
                        liwilines=nrows(liwiplot.graphics)+5
                        show(p.output, liwiplot); println()
                    end
                    p.ens_disp && (ensemblelines=show(p.output, p.e, progress=true))
                    p.src_disp && (println("MLE Model Sources:");srclines=show(p.output, p.top_m, nsrc=p.no_displayed_srcs, progress=true))
                    return lhlines + liwilines + ensemblelines + srclines
                end