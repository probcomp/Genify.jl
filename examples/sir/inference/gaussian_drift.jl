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
