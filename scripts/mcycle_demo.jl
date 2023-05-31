using DrWatson
@quickactivate :GPLab

using RDatasets
using Statistics
using CairoMakie
using DataFrames

function plot_pred(xtest, ypred)
    ymean = mean.(ypred)
    ystd = std.(ypred)
    yupper = ymean .+ (2 .* ystd)
    ylower = ymean .- (2 .* ystd)
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel="Time", ylabel="Accel")
    scatter!(ax, x[:, 1], y)
    lines!(ax, xtest[:, 1], ymean)
    band!(ax, xtest[:, 1], ylower, yupper; color=(:grey, 0.5))
    return fig
end

dat = dataset("MASS", "mcycle")
x = Matrix(dat[:, [:Times]])
y = dat[:, :Accel]

datr = combine(groupby(dat, :Times)) do sdf
    (Accel=(sdf[:, :Accel],),)
end
datr[!, :Accel] = first.(datr[:, :Accel])
xr = Matrix(datr[:, [:Times]])
yr = datr[:, :Accel]

xtest = reshape(LinRange(minimum(x), maximum(x), 500), :, 1)

gpr = GPRegressor()
gpfit = fit(gpr, x, y)
println(gpfit.params)
ypred = predict(gpfit, xtest)

save(plotsdir("mcycle_std.png"), plot_pred(xtest, ypred); resolution = (600, 400))

rgpr = RepGPRegressor()
rgpfit = fit(rgpr, xr, yr)
println(rgpfit.params)
yrpred = predict(rgpfit, xtest)

save(plotsdir("mcycle_rep.png"), plot_pred(xtest, yrpred); resolution = (600, 400))
