# [Probabilistic Programming in Thirty Seconds - Turing Quickstart](https://turing.ml/dev/docs/using-turing/quick-start)

# If you are already well-versed in probabilistic programming and just
# want to take a quick look at how Turing's syntax works or otherwise
# just want a model to start with, we have provided a Bayesian
# coin-flipping model to play with.

# This example can be run on however you have Julia installed (see Getting Started),
# but you will need to install the packages `Turing`` and `StatsPlots``
# if you have not done so already.

# This is an excerpt from a more formal example introducing probabilistic
# programming which can be found in Jupyter notebook form here or as part
# of the documentation website here.

# Import libraries.
using Turing, StatsPlots, Random

# Set the true probability of heads in a coin.
p_true = 0.5 # can also test with biased coin

# Iterate from having seen 0 observations to 100 observations.
Ns = 0:100

# Draw data from a Bernoulli distribution, i.e. draw heads or tails.
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# Declare our Turing model.
@model function coinflip(y)
    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1) # flat prior over 0..1

    # The number of observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 1000
ϵ = 0.05 # HMC arg ϵ::Float64 : The leapfrog step size to use.
τ = 10 # HMC arg n_leapfrog::Int : The number of leapfrog steps to use.

# Start sampling.
chain = sample(coinflip(data), HMC(ϵ, τ), iterations)

describe(chain)

# Plot a summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
histogram(chain[:p])
