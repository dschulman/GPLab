using DrWatson
@quickactivate :GPLab

using BenchmarkTools
using DataFrames
using RDatasets

mcycle = dataset("MASS", "mcycle")
x = Matrix(mcycle[:, [:Times]])
y = mcycle[:, :Accel]

fitstd = fit(GPRegressor(), x, y)
display(fitstd.lml)
display(fitstd.params)

std_trial = @benchmark fit(GPRegressor(), x, y)
display(std_trial)

mcycle_rep = combine(groupby(mcycle, :Times)) do sdf
    (Accel=(sdf[:, :Accel],),)
end
mcycle_rep[!, :Accel] = first.(mcycle_rep[:, :Accel])

xrep = Matrix(mcycle_rep[:, [:Times]])
yrep = mcycle_rep[:, :Accel]

fitrep = fit(RepGPRegressor(), xrep, yrep)
display(fitrep.lml)
display(fitrep.params)

rep_trial = @benchmark fit(RepGPRegressor(), xrep, yrep)
display(rep_trial)

display(ratio(median(std_trial), median(rep_trial)))
