### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 11220fee-00bb-11ee-228b-ada59a3eff31
using DrWatson

# ╔═╡ d1c4d64c-7096-48b9-b2de-3493c5d37521
@quickactivate

# ╔═╡ 53296287-b3e4-44ee-97bb-73a6661bb523
using Revise

# ╔═╡ 1c11c126-9562-4ddd-ba39-e882960ac90a
using GPLab

# ╔═╡ 4141e239-d097-45bb-8dc9-cc641efaa204
using RDatasets

# ╔═╡ 6a49b486-eeb6-42fa-b665-0b075b9a583a
using Statistics

# ╔═╡ 5218c999-b3cb-4f23-bbb8-d2759dbb1dfd
using CairoMakie

# ╔═╡ fe456507-6ed8-46ed-8aaf-ef82891d7fdd
using DataFrames

# ╔═╡ 6c6dcd91-a07d-44cc-99ae-d2e3d18d60f1
dat = dataset("MASS", "mcycle")

# ╔═╡ 3ce681b8-812d-458a-ad21-86dc9a0ef63f
x = Matrix(dat[:, [:Times]])

# ╔═╡ 8bd36264-831f-491a-b329-f1c6a1b43ca9
y = dat[:, :Accel]

# ╔═╡ b7f51dda-6464-429b-8700-1c667baa897d
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

# ╔═╡ 2f6abe25-fa2f-4cf8-88bd-86cb4565dc69
begin
	datr = combine(groupby(dat, :Times)) do sdf
	    (Accel=(sdf[:, :Accel],),)
	end
	datr[!, :Accel] = first.(datr[:, :Accel])
	datr
end

# ╔═╡ 9ce53bca-d2c3-4dd8-9996-211465e1ebef
xr = Matrix(datr[:, [:Times]])

# ╔═╡ 4cbd94da-a32d-4162-8073-d69f96f11d61
yr = datr[:, :Accel]

# ╔═╡ 61c222e8-2cb8-4319-ab48-80b7f8c0479c
xtest = reshape(LinRange(minimum(x), maximum(x), 500), :, 1)

# ╔═╡ 8d738cca-49ca-4b10-8a6a-1c94b004a12a
gpr = GPRegressor()

# ╔═╡ 743fd5a5-ba4b-4298-a650-9b8f0667b307
gpfit = fit(gpr, x, y)

# ╔═╡ b22e75ea-ecd2-462f-9a96-4bf552092f7d
gpfit.params

# ╔═╡ 1b8a1b43-65fc-479a-ab49-a39442630c3a
ypred = predict(gpfit, xtest)

# ╔═╡ 91d06279-6aa2-4718-9f80-31eb0e1b4cbb
plot_pred(xtest, ypred)

# ╔═╡ 08d84aa5-820f-4aa5-99b0-a87180eb9c7c
rgpr = RepGPRegressor()

# ╔═╡ 5ada7bc5-86d9-4839-86f0-32454f9559d7
rgpfit = fit(rgpr, xr, yr)

# ╔═╡ 84726cbf-5912-43e0-850d-f08dc3f8de5c
rgpfit.params

# ╔═╡ f9b33d81-f195-4ea5-bc72-c89019555a50
yrpred = predict(rgpfit, xtest)

# ╔═╡ 77310039-9b53-4f8d-b6bc-3615ba8a4340
plot_pred(xtest, yrpred)

# ╔═╡ b14136d4-15ad-4ce2-8427-6f18710a9389
lgpr = LaplaceGPRegressor(GPLab.Gaussian())

# ╔═╡ 4b8aa6ee-c054-4721-8bf5-e3c36b8e8a21
lgpfit = fit(lgpr, x, y)

# ╔═╡ 0833fe3b-05bc-4483-a2da-29a740c958b5
lgpfit.params

# ╔═╡ dcd3f78c-a841-4e2a-ab73-6087e15731f2
ylpred = predict(lgpfit, xtest)

# ╔═╡ 142f64c7-9788-44dd-a23b-535895f803dc
plot_pred(xtest, ylpred)

# ╔═╡ e9830e54-b107-4600-81c4-b80bdd25a0d7
lrgrp = LaplaceGPRegressor(GPLab.RepGaussian())

# ╔═╡ 9a379a3a-13e6-401b-a8d6-b2f02933ab34
lrgpfit = fit(lrgrp, xr, yr)

# ╔═╡ da878c54-1461-44b6-8e17-105b7c3bee60
lrgpfit.params

# ╔═╡ c2d7cc7a-81e1-48bb-9447-e96f275e485a
ylrpred = predict(lrgpfit, xtest)

# ╔═╡ b02ba126-3388-4c8a-be75-fe35650bbd99
plot_pred(xtest, ylrpred)

# ╔═╡ Cell order:
# ╠═11220fee-00bb-11ee-228b-ada59a3eff31
# ╠═d1c4d64c-7096-48b9-b2de-3493c5d37521
# ╠═53296287-b3e4-44ee-97bb-73a6661bb523
# ╠═1c11c126-9562-4ddd-ba39-e882960ac90a
# ╠═4141e239-d097-45bb-8dc9-cc641efaa204
# ╠═6a49b486-eeb6-42fa-b665-0b075b9a583a
# ╠═5218c999-b3cb-4f23-bbb8-d2759dbb1dfd
# ╠═fe456507-6ed8-46ed-8aaf-ef82891d7fdd
# ╠═b7f51dda-6464-429b-8700-1c667baa897d
# ╠═6c6dcd91-a07d-44cc-99ae-d2e3d18d60f1
# ╠═3ce681b8-812d-458a-ad21-86dc9a0ef63f
# ╠═8bd36264-831f-491a-b329-f1c6a1b43ca9
# ╠═2f6abe25-fa2f-4cf8-88bd-86cb4565dc69
# ╠═9ce53bca-d2c3-4dd8-9996-211465e1ebef
# ╠═4cbd94da-a32d-4162-8073-d69f96f11d61
# ╠═61c222e8-2cb8-4319-ab48-80b7f8c0479c
# ╠═8d738cca-49ca-4b10-8a6a-1c94b004a12a
# ╠═743fd5a5-ba4b-4298-a650-9b8f0667b307
# ╠═b22e75ea-ecd2-462f-9a96-4bf552092f7d
# ╠═1b8a1b43-65fc-479a-ab49-a39442630c3a
# ╠═91d06279-6aa2-4718-9f80-31eb0e1b4cbb
# ╠═08d84aa5-820f-4aa5-99b0-a87180eb9c7c
# ╠═5ada7bc5-86d9-4839-86f0-32454f9559d7
# ╠═84726cbf-5912-43e0-850d-f08dc3f8de5c
# ╠═f9b33d81-f195-4ea5-bc72-c89019555a50
# ╠═77310039-9b53-4f8d-b6bc-3615ba8a4340
# ╠═b14136d4-15ad-4ce2-8427-6f18710a9389
# ╠═4b8aa6ee-c054-4721-8bf5-e3c36b8e8a21
# ╠═0833fe3b-05bc-4483-a2da-29a740c958b5
# ╠═dcd3f78c-a841-4e2a-ab73-6087e15731f2
# ╠═142f64c7-9788-44dd-a23b-535895f803dc
# ╠═e9830e54-b107-4600-81c4-b80bdd25a0d7
# ╠═9a379a3a-13e6-401b-a8d6-b2f02933ab34
# ╠═da878c54-1461-44b6-8e17-105b7c3bee60
# ╠═c2d7cc7a-81e1-48bb-9447-e96f275e485a
# ╠═b02ba126-3388-4c8a-be75-fe35650bbd99
