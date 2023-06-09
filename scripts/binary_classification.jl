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
## Laplace GPR with Bernoulli-logit likelihood
"""

# ╔═╡ db4a19a8-3925-400b-abe8-8416fe797c51
llgpr = LaplaceGPRegressor(BernoulliLogitLikelihood())

# ╔═╡ 5ab386e8-55f0-49ec-8cb4-75c9046a91f7
llgpfit = fit(llgpr, reshape(x, :, 1), y; trace=true)

# ╔═╡ 70e1f05b-c3a1-4e37-a70f-dcd0524698c4
llgpfit.approx_lml

# ╔═╡ 4bdb8f22-5b21-49d6-8b92-f0f995095169
llgpfit.params

# ╔═╡ 0b999c07-42b2-4f5e-a2b1-60c810545956
logit_latent = reduce(vcat, mean.(predict_latent(llgpfit, reshape(Xgrid, :, 1))))

# ╔═╡ fab12822-612b-4236-9921-03b129f27277
logit_pred = mean.(predict(llgpfit, reshape(Xgrid, :, 1)))

# ╔═╡ 7b46a936-9350-4c92-b9a7-e2a7a0d33114
begin
	fig2, ax2, _ = scatter(x, y; label="Observations")
	lines!(ax2, x, ps; label="True probabilities")
	lines!(ax2, Xgrid, logistic.(logit_latent); linestyle=:dash, label="Expit of latent mean")
	lines!(ax2, Xgrid, logit_pred; linestyle=:dash, label="Posterior predictive")
	axislegend(ax2)
	fig2
end

# ╔═╡ 31044b73-0b89-4da6-b630-5f20633740b3
md"""
## Laplace GPR with Bernoulli-probit likelihood
"""

# ╔═╡ b81699ac-c7b5-4d65-82f4-dcafdbea3a23
lpgpr = LaplaceGPRegressor(BernoulliProbitLikelihood())

# ╔═╡ 16003b31-4a44-4b27-94f2-1850b69e4e4c
lpgpfit = fit(lpgpr, reshape(x, :, 1), y; trace=true)

# ╔═╡ 912fe451-2e7b-41ee-a5f1-0b69eba13b75
lpgpfit.approx_lml

# ╔═╡ 9f229b7b-bdad-4f46-a865-af7c60df77d6
lpgpfit.params

# ╔═╡ 4369fbcd-77b7-4bde-b0d0-8fe198c66096
probit_latent = reduce(vcat, mean.(predict_latent(lpgpfit, reshape(Xgrid, :, 1))))

# ╔═╡ ba227303-e6c3-475b-aded-61edfe9dd2f6
probit_pred = mean.(predict(lpgpfit, reshape(Xgrid, :, 1)))

# ╔═╡ e007fb25-7d73-4cd6-ae17-a6d6e57a7785
begin
	fig3, ax3, _ = scatter(x, y; label="Observations")
	lines!(ax3, x, ps; label="True probabilities")
	lines!(ax3, Xgrid, normcdf.(probit_latent); linestyle=:dash, label="Inv-probit of latent mean")
	lines!(ax3, Xgrid, probit_pred; linestyle=:dash, label="Posterior predictive")
	axislegend(ax3)
	fig3
end

# ╔═╡ 0d70c7d4-e16c-45fe-a440-4fecf497e29b
md"""
## Comparison
"""

# ╔═╡ 8ba4e44f-f405-41e6-a4ad-3277f79ab0e5
(logit=llgpfit.approx_lml, probit=lpgpfit.approx_lml)

# ╔═╡ e8eaa0f0-d446-4791-8f2c-e5b6a3dc8a24
begin
	fig4, ax4, _ = scatter(x, y; label="Observations")
	lines!(ax4, x, ps; label="True probabilities")
	lines!(ax4, Xgrid, logit_pred; linestyle=:dash, label="Logit")
	lines!(ax4, Xgrid, probit_pred; linestyle=:dash, label="Probit")
	axislegend(ax4)
	fig4
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
# ╠═70e1f05b-c3a1-4e37-a70f-dcd0524698c4
# ╠═4bdb8f22-5b21-49d6-8b92-f0f995095169
# ╠═0b999c07-42b2-4f5e-a2b1-60c810545956
# ╠═fab12822-612b-4236-9921-03b129f27277
# ╠═7b46a936-9350-4c92-b9a7-e2a7a0d33114
# ╠═31044b73-0b89-4da6-b630-5f20633740b3
# ╠═b81699ac-c7b5-4d65-82f4-dcafdbea3a23
# ╠═16003b31-4a44-4b27-94f2-1850b69e4e4c
# ╠═912fe451-2e7b-41ee-a5f1-0b69eba13b75
# ╠═9f229b7b-bdad-4f46-a865-af7c60df77d6
# ╠═4369fbcd-77b7-4bde-b0d0-8fe198c66096
# ╠═ba227303-e6c3-475b-aded-61edfe9dd2f6
# ╠═e007fb25-7d73-4cd6-ae17-a6d6e57a7785
# ╠═0d70c7d4-e16c-45fe-a440-4fecf497e29b
# ╠═8ba4e44f-f405-41e6-a4ad-3277f79ab0e5
# ╠═e8eaa0f0-d446-4791-8f2c-e5b6a3dc8a24
