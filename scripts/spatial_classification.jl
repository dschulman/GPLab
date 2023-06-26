### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 7efbb01e-1421-11ee-09f0-7590b2574fa3
using DrWatson

# ╔═╡ 46014d7c-97c4-4c0f-a32d-adcad97f0054
@quickactivate

# ╔═╡ 75b2dcc1-320e-4b62-913d-4aca44f36573
using GPLab

# ╔═╡ 4acac5ae-a004-4a02-93fb-cfbb259c3a26
using AbstractGPs

# ╔═╡ e08237d6-1594-4f8e-b985-4e09cc26e48d
using Statistics

# ╔═╡ e9bf0218-8504-4ac2-9220-37e6995af9b2
using RDatasets

# ╔═╡ 3cc54cea-59c7-4a17-b719-dd63656a8cd9
using CairoMakie

# ╔═╡ 21d3c7d0-d5ec-4dcb-96fa-337627a25dc5
md"""
Reproduction of <https://research.cs.aalto.fi//pml/software/gpstuff/demo_classific.shtml> (Laplace approximation only, no EP or MCMC).
"""

# ╔═╡ 0710ea3c-daa3-4a25-bf8d-840a5dc4110a
dat = dataset("MASS", "synth.tr")

# ╔═╡ a2a905fa-9e0e-4e77-9d48-94be176891d9
xy = Matrix(dat[!,[:XS,:YS]])

# ╔═╡ 70f0bff2-2eb2-48b3-a6c0-acf9e9c09e2e
z = dat[!,:YC]

# ╔═╡ 8ab60b9f-9662-4c32-bff3-2d17b7022906
scatter(xy[:,1], xy[:,2], color=z)

# ╔═╡ 85c14bc8-8967-456e-a06b-caf0b6b1353a
gp = LaplaceGPRegressor(BernoulliLogitLikelihood(), SqExponentialKernel())

# ╔═╡ 84ffe7c1-db28-43be-8c4c-281312bf1b08
gpfit = fit(gp, xy, z; trace=true)

# ╔═╡ 7867f7b3-ec92-4d06-af24-9013556dcf08
xtest = LinRange(-1.5, 1.0, 100)

# ╔═╡ 19096e1f-23b2-4958-b927-b76f748726f3
ytest = LinRange(-0.5, 1.5, 100)

# ╔═╡ 1db0af53-c0e4-4a76-89ba-1c01a3dfdeb2
xytest = reduce(vcat, ([xt, yt]' for xt in xtest for yt in ytest))

# ╔═╡ 1706150d-7797-4ae5-bc43-7a235b3676c9
zpred = mean.(predict(gpfit, xytest))

# ╔═╡ 2b561b02-78fd-4476-b8bc-98b12333f5f0
begin
	fig, ax, _ = scatter(xy[:,1], xy[:,2], color=z)
	contour!(
		ax, xytest[:,1], xytest[:,2], zpred ;
		colormap = :diverging_bkr_55_10_c35_n256,
		colorrange = ((0, 1)),
		linewidth = 2,
		levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
		labels = true
	)
	fig
end

# ╔═╡ Cell order:
# ╠═21d3c7d0-d5ec-4dcb-96fa-337627a25dc5
# ╠═7efbb01e-1421-11ee-09f0-7590b2574fa3
# ╠═46014d7c-97c4-4c0f-a32d-adcad97f0054
# ╠═75b2dcc1-320e-4b62-913d-4aca44f36573
# ╠═4acac5ae-a004-4a02-93fb-cfbb259c3a26
# ╠═e08237d6-1594-4f8e-b985-4e09cc26e48d
# ╠═e9bf0218-8504-4ac2-9220-37e6995af9b2
# ╠═0710ea3c-daa3-4a25-bf8d-840a5dc4110a
# ╠═a2a905fa-9e0e-4e77-9d48-94be176891d9
# ╠═70f0bff2-2eb2-48b3-a6c0-acf9e9c09e2e
# ╠═3cc54cea-59c7-4a17-b719-dd63656a8cd9
# ╠═8ab60b9f-9662-4c32-bff3-2d17b7022906
# ╠═85c14bc8-8967-456e-a06b-caf0b6b1353a
# ╠═84ffe7c1-db28-43be-8c4c-281312bf1b08
# ╠═7867f7b3-ec92-4d06-af24-9013556dcf08
# ╠═19096e1f-23b2-4958-b927-b76f748726f3
# ╠═1db0af53-c0e4-4a76-89ba-1c01a3dfdeb2
# ╠═1706150d-7797-4ae5-bc43-7a235b3676c9
# ╠═2b561b02-78fd-4476-b8bc-98b12333f5f0
