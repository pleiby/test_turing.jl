print("It's alive!")

using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
  s² ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s²))
  x ~ Normal(m, sqrt(s²)) # assume independent data are normally distributed
  y ~ Normal(m, sqrt(s²))
end

# Run sampler, collect results
# Return 1000 samples from the model with the Markov chain Monte Carlo sampler HMC.
chn = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)

# Summarise results

describe(chn)

# Plot and save results
p = plot(chn)
savefig("gdemo-plotv1.png")

print("Done")