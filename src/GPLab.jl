module GPLab

using AbstractGPs
using ChainRulesCore
using Distributions
using LineSearches
using LinearAlgebra
using Optim
using ParameterHandling
using SpecialFunctions
using Statistics
using StatsBase
using StatsFuns
using StructArrays
using Zygote

export GPRegressor, RepGPRegressor, LaplaceGPRegressor, fit, predict_latent, predict,
    Likelihood, SimpleLikelihood, Replicate,
    BernoulliLogitLikelihood, BernoulliProbitLikelihood,
    GaussianLikelihood,
    TLikelihood

include("util.jl")
include("likelihood.jl")
include("likelihoods/bernoulli.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/t.jl")
include("std_gpr.jl")
include("rep_gpr.jl")
include("laplace_gpr.jl")

end
