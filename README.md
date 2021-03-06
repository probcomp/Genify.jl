# Genify.jl

Automatically transforms Julia methods to [Gen](https://www.gen.dev/) functions via staged compilation. All calls to random primitives like `Base.rand` or `StatsBase.sample` are
converted to samples from Gen distributions, and address names are generated
automatically. Gen-ification is applied recursively by default, tracing all nested sub-routines with stochastic behavior.

## Installation

At the Julia REPL, press `]` to enter the package manager, then run:
```
add https://github.com/probcomp/Genify.jl.git
```

## Usage

`Genify.jl` allows one to convert stochastic functions from plain Julia into Gen, enabling programmable inference via the manipulation of internal random variables. The user-level function for transforming Julia methods is `genify`, documented below:

> `genify(fn, arg_types...; kwargs...)` or `genify(options, fn, arg_types...)`
>
> Transforms a Julia method into a dynamic Gen function.
>
> **Arguments:**
> - `fn`: a `Function`, `Type` constructor, or (if the second form is used)
    any other callable object.
> - `arg_types`: The types of the arguments for the method to be transformed.
>
> **Keyword Arguments:**
> - `recurse::Bool=true`: recursively `genify` methods called by `fn` if true.
> - `useslots::Bool=true`: if true, use slot (i.e. variable) names as trace
    addresses where possible.
> - `naming::Symbol=:static`: scheme for generating address names, defaults to
    static generation at compile time. Use `:manual` for user-specified
    addresses (e.g., `rand(:z, Normal(0, 1))`)
> - `options`: the above options can also be provided as parameters in an
    `Options` struct, or as a `Symbol` from the list of named
    option sets overriding any other values specified:
>   - `:minimal` corresponds to `recurse=false, useslots=false, naming=:static`
>   - `:default` corresponds to `recurse=true, useslots=true, naming=:static`
>   - `:manual` corresponds to `recurse=true, useslots=false, naming=:manual`

`genify` should only be used at the REPL, or at the top-level of a script or module. A memoized version, `genified`, is designed to be used within other Gen functions.

## Example

Suppose we have the following simulator of a noisy weighing scale, written in Julia, which samples the mass of the weighed object, and simulates `n` measurements:

```julia
using Distributions: Normal

function scale(n::Int)
    mass = rand(Normal(5, 1))
    obs = measure(mass, n)
end

function measure(mass::Real, n::Int)
    obs = Float64[]
    for i in 1:n
        m = rand(Normal(mass, 2))
        push!(obs, m)
    end
    return obs
end
```

We can convert the `scale` method into generative function `genscale` using `genify`. Note that we need to specify both the name of `scale`, as well as the types of its arguments:

```julia
using Gen, Genify
genscale = genify(scale, Int)

julia> typeof(genscale)
Gen.DynamicDSLFunction{Any}
```

We can now treat this as a Gen function that supports the [generative function interface](https://www.gen.dev/dev/ref/gfi/). For example, we can simulate a trace (note that the first run will be a little slow, because transformation occurs during just-in-time compilation):

```julia
julia> trace = simulate(genscale, (3,));
julia> get_choices(trace)
├── :mass : 5.69796235005049
└── :obs
    └── :m
        ├── 1 : 4.267277635953466
        ├── 2 : 4.499125425390254
        └── 3 : 4.34687310812849
```

Notice that addresses for each random variable have been generated automatically from the source code, including variables that are sampled within loops.

We can also generate a trace while constraining observations:
```julia
julia> observations = choicemap(((:obs => :m => i, 6) for i in 1:3)...);
julia> trace, w = generate(genscale, (3), observations);
julia> get_choices(trace)
├── :mass : 5.843712561769999
└── :obs
    └── :m
        ├── 1 : 6.0
        ├── 2 : 6.0
        └── 3 : 6.0
```

This allows us to perform inference over the latent variables, e.g. via importance sampling:
```julia
observations = choicemap(((:obs => :m => i, 4) for i in 1:10)...)
trs, ws, _ = importance_sampling(genscale, (10,), observations, 100)
mass_est = sum(tr[:mass] * exp(w) for (tr, w) in zip(trs, ws))

julia> mass_est
4.288428017814824
```

For a more detailed tutorial, see the [`tutorial`](tutorial) directory. For more examples of programmable inference over transformed Julia code, see the [`examples`](examples) directory.
