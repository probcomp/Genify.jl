# Gaussian drift proposal
@gen function gaussian_drift(tr::Trace, params=Dict(:β => (0.3, 0.01, Inf)))
    for (addr, distargs) in params
        if distargs isa Tuple && length(distargs) == 3
            sigma, lb, ub = distargs
            {addr} ~ trunc_normal(tr[addr], sigma, lb, ub)()
        else
            sigma = distargs
            {addr} ~ normal(tr[addr], sigma)()
        end
    end
end

# Guess beta from the data from initial infection counts
@gen function init_beta(obs::ChoiceMap, t::Int)
    params = create_params()
    cities = 1:length(params[:Ns])
    infected_0 = sum(params[:Is])
    infected_t = sum(obs[:step => t => :obs => :infected => c] for c in cities)
    β_hat = (infected_t/infected_0) ^ (1/t) - 1
    β ~ trunc_normal(β_hat, 0.1, 0, Inf)()
end
