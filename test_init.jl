print("It's alive - Turing.jl Hello World!")


using Turing
using StatsPlots

# [Hello World in Turing](https://turing.ml/stable/)

# Define a simple Normal model gdemo with unknown mean and variance.
@model function gdemo(x, y)
  s² ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s²))
  x ~ Normal(m, sqrt(s²)) # assume independent data are normally distributed
  y ~ Normal(m, sqrt(s²)) # x and y distrib with same unknown mean m, var s²
end

# Notes re dists:
# Note: "chief use of the **inverse gamma distribution** is in Bayesian statistics,
# where the distribution arises as the marginal posterior distribution
# for the unknown variance of a normal distribution, if an uninformative
# prior is used, and as an analytically tractable conjugate prior,
# if an informative prior is required." [Inverse Gamma, Wikipedia](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)

# Note: "In Bayesian probability theory, if the posterior distribution p(θ | x)
# is in the same probability distribution family as the prior probability
# distribution p(θ), the prior and posterior are then called
# **conjugate distributions**" [Conjugate prior, wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior)

# Note: "The **Shannon entropy** of the random variable $X$ is defined as
# ... the expected information content of [a] measurement of $X$."
# The "self-information" of an observation $X=x$ is a measure of its
# surprise, i.e. for $Prob(X=x) = p_X(x)$ then info is $I(x) = -log(p_X(x))$.
# So the Shannon Entropy of r.v. X,  $H(X)$ is the functional
# $H(X) = - \sum_x p_X(x) log(p_X(x))$

# Run sampler, collect results
# Return 1000 samples from the model with the Markov Chain Monte Carlo
#  sampler HMC.
chn = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)

# Summarise results

describe(chn)

# Plot and save results
p = plot(chn)
savefig("gdemo-plotv1.png")

print("Done")