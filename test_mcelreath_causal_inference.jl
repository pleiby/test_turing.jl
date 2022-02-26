# Turing.jl for Causal Inference Model

# From julialang discourse

# [Turing.jl for Causal Inference Model](https://discourse.julialang.org/t/turing-jl-for-causal-inference-model/75650) danielw2904
# > I am trying to translate the model from McElreath’s Causal Inference workshop
#  ([youtube Science Before Statistics](https://www.youtube.com/watch?v=KNPYUVmY3NM), 
#  [github Causal Salad](https://github.com/rmcelreath/causal_salad_2021)) 
#  to Turing.jl and while the results seem to be the same
#  I get much worse performance with `Turing`` than `Stan` (3 sec. vs 110 sec.)


## Rethinking Version
R"""
set.seed(1908)
N <- 200 # number of pairs
U <- rnorm(N) # simulate confounds
# birth order and family sizes
B1 <- rbinom(N,size=1,prob=0.5) # 50% first borns
M <- rnorm( N , 2*B1 + U )
B2 <- rbinom(N,size=1,prob=0.5)
D <- rnorm( N , 2*B2 + U + 0*M ) # change the 0 to turn on causal influence of mom
library(rethinking)
library(cmdstanr)
dat <- list(N=N,M=M,D=D,B1=B1,B2=B2)
set.seed(1908)
flbi <- ulam(
    alist(
        # mom model
            M ~ normal( mu , sigma ),
            mu <- a1 + b*B1 + k*U[i],
        # daughter model
            D ~ normal( nu , tau ),
            nu <- a2 + b*B2 + m*M + k*U[i],
        # B1 and B2
            B1 ~ bernoulli(p),
            B2 ~ bernoulli(p),
        # unmeasured confound
            vector[N]:U ~ normal(0,1),
        # priors
            c(a1,a2,b,m) ~ normal( 0 , 0.5 ),
            c(k,sigma,tau) ~ exponential( 1 ),
            p ~ beta(2,2)
    ), data=dat , chains=4 , cores=4 , iter=2000 , cmdstan=TRUE )
posterior <- extract.samples(flbi)
""";
posterior_R = @rget(posterior);
dat_R = @rget(dat);

@model function mom(N, M, D, B1, B2)
    p ~ Beta(2, 2)
    k ~ Exponential(1)
    σ ~ Exponential(1)
    τ ~ Exponential(1)
    a1 ~ Normal(0, 0.5)
    a2 ~ Normal(0, 0.5)
    b ~ Normal(0, 0.5)
    m ~ Normal(0, 0.5)
    U ~ filldist(Normal(0, 1), N)
    B1 ~ Bernoulli(p)
    B2 ~ Bernoulli(p)

    ν = a2 .+ b * B2 + m * M + k * U
    D .~ Normal.(ν, τ)

    μ = a1 .+ b * B1 + k * U
    M .~ Normal.(μ, σ)
end


Turing.setrdcache(true)
Turing.setadbackend(:reversediff)

flbi = sample(mom(Int(dat_R[:N]), dat_R[:M], dat_R[:D], dat_R[:B1], dat_R[:B2]),
    NUTS(1000, 0.65),
    MCMCThreads(),
    2000, 4)