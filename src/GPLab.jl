module GPLab

using AbstractGPs
using LineSearches
using LinearAlgebra
using Optim
using ParameterHandling
using Statistics
using StatsFuns
using Zygote

export GPRegressor, RepGPRegressor, fit

include("util.jl")
include("std_regression.jl")
include("rep_regression.jl")

end
