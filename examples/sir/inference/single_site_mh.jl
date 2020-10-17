# Extracts all choice addresses from a choice map or trace
function all_addresses(choices::Gen.ChoiceMapNestedView,
                       exclude::Selection=EmptySelection())
    addrs = []
    choices = sort(collect(choices), by=first)
    for (k, v) in choices
        if v isa Gen.ChoiceMapNestedView
            append!(addrs, [Pair(k, a) for a in all_addresses(v, exclude[k])])
        elseif exclude[k] != AllSelection()
            push!(addrs, k)
        end
    end
    return addrs
end
all_addresses(choices::ChoiceMap, exclude::Selection=EmptySelection()) =
    all_addresses(nested_view(choices), exclude)
all_addresses(trace::Trace, exclude::Selection=EmptySelection()) =
    all_addresses(get_choices(trace), exclude)

# Single site Metropolis Hastings
function single_site_mh(T::Int, observations::ChoiceMap, n_iters::Int,
                        tracked_vars=[:β], obs_noise=5.0)
    trs, scores = Vector{Trace}(undef, n_iters) ,Vector{Float64}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    # Smart initialization of beta from the data
    init_choices, _, _ = propose(init_beta, (observations, min(T, 10)))
    observations = merge(observations, init_choices)
    trace, _ = generate(bayesian_sir, (T, obs_noise), observations)
    trs[1], scores[1] = trace, get_score(trace)
    data[1, tracked_vars] = [trace[v] for v in tracked_vars]
    # Extract addresses for all unobserved variables
    obs_addrs = Gen.select([:step => t => :obs for t in 1:T]...)
    latent_addrs = all_addresses(trace, obs_addrs)
    # Sequentially regenerate each address in the original trace
    for i in 2:n_iters
        @showprogress for addr in latent_addrs
            trace, _ = mh(trace, Gen.select(addr))
        end
        trs[i], scores[i] = trace, get_score(trace)
        data[i, tracked_vars] = [trace[v] for v in tracked_vars]
    end
    return trs, scores, data
end
