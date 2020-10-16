# Extracts all choice addresses from a choice map or trace
function all_addresses(choices::Gen.ChoiceMapNestedView)
    addrs = []
    choices = sort(collect(choices), by=first)
    for (k, v) in choices
        if v isa Gen.ChoiceMapNestedView
            append!(addrs, [Pair(k, a) for a in all_addresses(v)])
        else
            push!(addrs, k)
        end
    end
    return addrs
end
all_addresses(choices::ChoiceMap) = all_addresses(nested_view(choices))
all_addresses(trace::Trace) = all_addresses(get_choices(trace))

# Single site Metropolis Hastings
function single_site_mh(T::Int, observations::ChoiceMap, n_iters::Int,
                        tracked_vars=[:β], obs_noise=5.0)
    trs, scores = Vector{Trace}(undef, n_iters) ,Vector{Float64}(undef, n_iters)
    data = DataFrame(fill(Float64, length(tracked_vars)), tracked_vars, n_iters)
    # Generate initial trace
    trace, _ = generate(bayesian_sir, (T, obs_noise), observations)
    trs[1], scores[1] = trace, get_score(trace)
    data[i, tracked_vars] = [trace[v] for v in tracked_vars]
    # Extract addresses for all unobserved variables
    latent_addrs = setdiff(all_addresses(trace), all_addresses(observations))
    # Sequentially regenerate each address in the original trace
    for i in 2:n_iters
        for addr in latent_addrs
            trace, _ = mh(trace, Gen.select(addr))
        end
        trs[i], scores[i] = trace, get_score(trace)
        data[i, tracked_vars] = [trace[v] for v in tracked_vars]
    end
    return trs, scores, data
end
