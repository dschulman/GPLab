### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 0e615054-3b7c-467b-b68b-9b9ececf8e22
using DrWatson

# ╔═╡ 76db0933-bba0-4e07-9c02-82e20ad1ee5b
@quickactivate

# ╔═╡ fe823cb6-19dc-11ee-32fc-d9aab3d79925
using Revise

# ╔═╡ fbc731a4-5388-40c6-806c-1a168d9f2e9b
using GPLab

# ╔═╡ e283f57d-d871-418b-8fce-99a34e8109ca
using Distributions: Gamma

# ╔═╡ f5fcfe82-afa1-4d0c-a669-08787b1019c8
using Distributions: shape

# ╔═╡ f4c338c2-0b59-4c52-ba7a-083b57c7f59f
using Statistics

# ╔═╡ b04e0061-d66d-4def-8b6f-ab6b6299953f
using Random

# ╔═╡ 79e43177-8b53-4f5c-af44-d20a7807d6f4
using CairoMakie

# ╔═╡ d12b9986-5e79-4196-9b6d-478af40102ea
import Distributions

# ╔═╡ 3d228bfa-93f8-4e38-b3e3-626d0a9e4591
x = reshape(LinRange(-1, 1, 100), :, 1)

# ╔═╡ 0e5243e0-38f6-4a32-a1d1-0af12330044a
m = exp.(sin.(vec(x) .* 2) .* 0.5)

# ╔═╡ 34134384-fc5a-4032-aa07-a6f621233509
k = exp.(cos.(vec(x) .* 2)) .* 2

# ╔═╡ 1ebcf454-06ec-41c3-8449-6f9f7c4b1711
y = let rng = Xoshiro(12345)
	rand.(rng, Gamma.(k, m ./ k))
end

# ╔═╡ 4919a742-95e0-44ed-ae56-52d7a9b59e8c
scatter(vec(x), y)

# ╔═╡ 4c8227fd-89c0-4b36-a5b3-7cf7f6075eb7
gpr = LaplaceGPRegressor(GammaLikelihood())

# ╔═╡ 715ffffc-9bb9-4eda-872b-ee5ad31ac8f8
gpfit = fit(gpr, x, y; trace=true)

# ╔═╡ e021bc55-8780-41fd-9edc-b1cae930816d
lmean, lvar = predict_latent(gpfit, x)

# ╔═╡ 3f0d0084-2884-4c62-8a25-9edc2e7ff444
ypred = predict(gpfit, x)

# ╔═╡ 2adf9ae3-e23a-4a84-a3d2-3b3257f4197a
let fig = Figure()
	ax = Axis(fig[1,1], title="Mean (μ)")
	lines!(ax, vec(x), m, label="True")
	lines!(ax, vec(x), exp.(lmean[:,1]), linestyle=:dash, label="Latent")
	lines!(ax, vec(x), mean.(ypred), linestyle=:dashdot, label="Postpred")
	axislegend(ax)
	fig
end

# ╔═╡ 9c00b0bd-b8f5-40dd-a386-b8aea457ac9d
let fig = Figure()
	ax = Axis(fig[1,1], title="Shape (k)")
	lines!(ax, vec(x), k, label="True")
	lines!(ax, vec(x), exp.(lmean[:,2]), linestyle=:dash, label="Latent")
	lines!(ax, vec(x), shape.(ypred), linestyle=:dashdot, label="Postpred")
	axislegend(ax)
	fig
end

# ╔═╡ Cell order:
# ╠═fe823cb6-19dc-11ee-32fc-d9aab3d79925
# ╠═0e615054-3b7c-467b-b68b-9b9ececf8e22
# ╠═76db0933-bba0-4e07-9c02-82e20ad1ee5b
# ╠═fbc731a4-5388-40c6-806c-1a168d9f2e9b
# ╠═d12b9986-5e79-4196-9b6d-478af40102ea
# ╠═e283f57d-d871-418b-8fce-99a34e8109ca
# ╠═f5fcfe82-afa1-4d0c-a669-08787b1019c8
# ╠═f4c338c2-0b59-4c52-ba7a-083b57c7f59f
# ╠═b04e0061-d66d-4def-8b6f-ab6b6299953f
# ╠═79e43177-8b53-4f5c-af44-d20a7807d6f4
# ╠═3d228bfa-93f8-4e38-b3e3-626d0a9e4591
# ╠═0e5243e0-38f6-4a32-a1d1-0af12330044a
# ╠═34134384-fc5a-4032-aa07-a6f621233509
# ╠═1ebcf454-06ec-41c3-8449-6f9f7c4b1711
# ╠═4919a742-95e0-44ed-ae56-52d7a9b59e8c
# ╠═4c8227fd-89c0-4b36-a5b3-7cf7f6075eb7
# ╠═715ffffc-9bb9-4eda-872b-ee5ad31ac8f8
# ╠═e021bc55-8780-41fd-9edc-b1cae930816d
# ╠═3f0d0084-2884-4c62-8a25-9edc2e7ff444
# ╠═2adf9ae3-e23a-4a84-a3d2-3b3257f4197a
# ╠═9c00b0bd-b8f5-40dd-a386-b8aea457ac9d
