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
include("std_gpr.jl")
include("rep_gpr.jl")

end
