module GPLab

using AbstractGPs
using ChainRulesCore
using Distributions
using LineSearches
using LinearAlgebra
using Optim
using ParameterHandling
using Statistics
using StatsFuns
using Zygote

export GPRegressor, RepGPRegressor, LaplaceGPRegressor, fit, predict_latent, predict,
    Likelihood, SimpleLikelihood, Replicate,
    GaussianLikelihood

include("util.jl")
include("likelihood.jl")
include("likelihoods/gaussian.jl")
include("std_gpr.jl")
include("rep_gpr.jl")
include("laplace_gpr.jl")

end
