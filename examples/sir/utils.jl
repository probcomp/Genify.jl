function case_count_obs(trace::Trace)
    T = get_args(trace)[1]
    addrs = Gen.select([:step => t => :obs => :infected for t in 1:T]...,
                       [:step => t => :obs => :recovered for t in 1:T]...)
    observations = Gen.get_selected(get_choices(trace), addrs)
    return observations
end

function location_obs(trace::Trace; frac::Float64=0.5)
    model, T = get_retval(trace), get_args(trace)[1]
    tracked = 1:Int(round(frac * nagents(model)))
    addrs = Gen.select([:step => t => :obs => :location => agt
                        for t in 1:T for agt in tracked]...)
    observations = Gen.get_selected(get_choices(trace), addrs)
    return observations
end

function migration_obs(trace::Trace)
    model, T = get_retval(trace), get_args(trace)[1]
    m_addrs = Gen.select(
        [:step => t => :agents => :step! => :agent_step! => 1 =>
         agt => :migrate! => :m for t in 1:T for agt in 1:nagents(model)]...
    )
    observations = Gen.get_selected(get_choices(trace), m_addrs)
    return observations
end

function obs_dataframe(tr::Trace)
    model, T = get_retval(tr), get_args(tr)[1]
    data = reduce(vcat, tr[:step => t => :obs]' for t in 1:T)
    cnames = reduce(vcat, [[Symbol(:I, i), Symbol(:R, i)] for i in 1:model.C])
    return DataFrame(data, cnames)
end

function plot_obs!(tr::Trace, cities=nothing, legend=false; kwargs...)
    model = get_retval(tr)
    cities = isnothing(cities) ? collect(1:model.C) : cities
    cnames = reduce(vcat, [[Symbol(:I, i), Symbol(:R, i)] for i in cities])
    df = obs_dataframe(tr)[:, cnames]
    if !haskey(kwargs, :ls)
        kwargs = Dict{Symbol,Any}(kwargs...)
        kwargs[:ls] = reduce(hcat, [[:solid :dash] for i in cities])
    end
    labels = legend == false ? fill("", length(cnames)) : names(df)
    plot!(Matrix(df); lw=2, labels=permutedims(labels),
          legend=legend, ylims=[0, maximum(model.Ns)*1.5], kwargs...)
end

function plot_obs(tr::Trace, cities=nothing, legend=:topleft; kwargs...)
    plot(); plot_obs!(tr, cities, legend; kwargs...)
end

function plot_obs!(trs::AbstractVector{<:Trace}, cities=nothing; kwargs...)
    for tr in trs plot_obs!(tr, cities, false; kwargs...) end
    return plot!()
end

function plot_obs!(trs::AbstractVector{<:Trace}, ws::AbstractVector{Float64},
                  cities=nothing; kwargs...)
    max_w = maximum(ws)
    for (tr, w) in zip(trs, ws)
        plot_obs!(tr, cities, false; alpha=w/max_w, kwargs...)
    end
    return plot!()
end
