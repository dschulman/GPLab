using DrWatson
@quickactivate :GPLab

using RDatasets

dat = dataset("MASS", "mcycle")
x = Matrix(dat[:, [:Times]])
y = dat[:, :Accel]

gpr = GPRegressor()
gpfit = fit(gpr, x, y)
println(gpfit.params)

xtest = reshape(LinRange(minimum(x), maximum(x), 500), :, 1)
ypred = predict(gpfit, xtest)

using Statistics
ymean = mean.(ypred)
ystd = std.(ypred)
yupper = ymean .+ (2 .* ystd)
ylower = ymean .- (2 .* ystd)

using CairoMakie

fig = Figure()
ax = Axis(fig[1,1]; xlabel="Time", ylabel="Accel")
scatter!(ax, x[:, 1], y)
lines!(ax, xtest[:, 1], ymean)
band!(ax, xtest[:, 1], ylower, yupper; color=(:grey, 0.5))
save(plotsdir("mcycle_std.png"), fig; resolution = (600, 400))
