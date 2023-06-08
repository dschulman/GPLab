### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 87bfe482-3dcc-4492-931c-0d69198b5018
using DrWatson

# ╔═╡ 6c0155e0-dc93-49e8-986b-af1d18bbe5e9
@quickactivate

# ╔═╡ 3b031ed4-dc78-48ed-b3d0-c27dde938484
using Revise

# ╔═╡ 59866755-793c-4a21-b0bd-254a2d563314
using GPLab

# ╔═╡ 4bef72ad-8443-4a17-ba16-c9936f022ba8
using AbstractGPs

# ╔═╡ 544063ba-97be-412d-a347-b2934bc7929c
using Random

# ╔═╡ f796a9f7-9bda-49de-baa1-60e144d2d0e8
using StatsFuns

# ╔═╡ c461869a-c639-4deb-b636-66f4773e70f7
using Distributions: Bernoulli

# ╔═╡ 135a1d7d-e8df-4eda-978b-2c5a1fa40451
using Statistics

# ╔═╡ 77c0fa20-d8e8-460a-bbe2-3020e015d707
using CairoMakie

# ╔═╡ 5eb9f144-0580-11ee-243f-fbc0e75aeabb
md"""
A demo reimplementation of <https://juliagaussianprocesses.github.io/ApproximateGPs.jl/dev/examples/c-comparisons/>
"""

# ╔═╡ e12caa5c-8493-49b9-ab75-0227bc4438aa
md"""
## Generate Training Data
"""

# ╔═╡ 88e4ee9c-21af-4dd3-9ac5-8562186d3862
Xgrid = -4:0.1:29  # for visualization

# ╔═╡ 7b6f5dfb-6f6c-477f-aa57-20876ee86545
x = range(0, 23.5; length=48)  # training inputs

# ╔═╡ c69964e6-e0d6-499a-b18a-76a8c2033d1d
f(x) = 3 * sin(10 + 0.6x) + sin(0.1x) - 1  # latent function

# ╔═╡ 9c5cd00f-3a6c-40e2-9b56-2e9b4ced5de3
fs = f.(x)  # latent function values at training inputs

# ╔═╡ 0092bba9-5d4b-41ff-99ca-601d758056bc
ps = logistic.(fs)

# ╔═╡ d373c349-0011-4673-a7b9-f167562ab3a7
begin
	rng = Xoshiro(1234)
	y = rand.(rng, Bernoulli.(ps))
end

# ╔═╡ 3ec9cfbc-e396-4065-abad-b48490bc93cd
begin
	fig, ax, _ = scatter(x, y; label="Observations")
	lines!(ax, x, ps; label="True probabilities")
	axislegend(ax)
	fig
end

# ╔═╡ 494c0caf-da3a-41c5-8b1c-7437fd5f4259
md"""
## Fit a Laplace GPR with Bernoulli-logit likelihood
"""

# ╔═╡ db4a19a8-3925-400b-abe8-8416fe797c51
lgpr = LaplaceGPRegressor(BernoulliLogitLikelihood())

# ╔═╡ 5ab386e8-55f0-49ec-8cb4-75c9046a91f7
lgpfit = fit(lgpr, reshape(x, :, 1), y; trace=true)

# ╔═╡ 8a4533a2-57ef-4ce5-b9f9-10f2ea5b2598
md"""
## Results
"""

# ╔═╡ 4bdb8f22-5b21-49d6-8b92-f0f995095169
lgpfit.params

# ╔═╡ 0b999c07-42b2-4f5e-a2b1-60c810545956
fpred = predict_latent(lgpfit, reshape(Xgrid, :, 1))

# ╔═╡ 0834696b-f4d0-4f0f-9b66-748e08186779
fpred_mean = reduce(vcat, mean.(fpred))

# ╔═╡ 8fe758bd-7a43-485b-9bb8-6e38edff94e3
fpred_std = sqrt.(reduce(vcat, var.(fpred)))

# ╔═╡ fab12822-612b-4236-9921-03b129f27277
ypred = mean.(predict(lgpfit, reshape(Xgrid, :, 1)))

# ╔═╡ 7b46a936-9350-4c92-b9a7-e2a7a0d33114
begin
	fig2, ax2, _ = scatter(x, y; label="Observations")
	lines!(ax2, x, ps; label="True probabilities")
	lines!(Xgrid, logistic.(fpred_mean); linestyle=:dash, label="Expit of latent mean")
	lines!(ax2, Xgrid, ypred; linestyle=:dash, label="Posterior predictive")
	axislegend(ax2)
	fig2
end

# ╔═╡ Cell order:
# ╠═5eb9f144-0580-11ee-243f-fbc0e75aeabb
# ╠═87bfe482-3dcc-4492-931c-0d69198b5018
# ╠═6c0155e0-dc93-49e8-986b-af1d18bbe5e9
# ╠═3b031ed4-dc78-48ed-b3d0-c27dde938484
# ╠═59866755-793c-4a21-b0bd-254a2d563314
# ╠═4bef72ad-8443-4a17-ba16-c9936f022ba8
# ╠═544063ba-97be-412d-a347-b2934bc7929c
# ╠═f796a9f7-9bda-49de-baa1-60e144d2d0e8
# ╠═c461869a-c639-4deb-b636-66f4773e70f7
# ╠═135a1d7d-e8df-4eda-978b-2c5a1fa40451
# ╠═77c0fa20-d8e8-460a-bbe2-3020e015d707
# ╠═e12caa5c-8493-49b9-ab75-0227bc4438aa
# ╠═88e4ee9c-21af-4dd3-9ac5-8562186d3862
# ╠═7b6f5dfb-6f6c-477f-aa57-20876ee86545
# ╠═c69964e6-e0d6-499a-b18a-76a8c2033d1d
# ╠═9c5cd00f-3a6c-40e2-9b56-2e9b4ced5de3
# ╠═0092bba9-5d4b-41ff-99ca-601d758056bc
# ╠═d373c349-0011-4673-a7b9-f167562ab3a7
# ╠═3ec9cfbc-e396-4065-abad-b48490bc93cd
# ╠═494c0caf-da3a-41c5-8b1c-7437fd5f4259
# ╠═db4a19a8-3925-400b-abe8-8416fe797c51
# ╠═5ab386e8-55f0-49ec-8cb4-75c9046a91f7
# ╠═8a4533a2-57ef-4ce5-b9f9-10f2ea5b2598
# ╠═4bdb8f22-5b21-49d6-8b92-f0f995095169
# ╠═0b999c07-42b2-4f5e-a2b1-60c810545956
# ╠═0834696b-f4d0-4f0f-9b66-748e08186779
# ╠═8fe758bd-7a43-485b-9bb8-6e38edff94e3
# ╠═fab12822-612b-4236-9921-03b129f27277
# ╠═7b46a936-9350-4c92-b9a7-e2a7a0d33114
