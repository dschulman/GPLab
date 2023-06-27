### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 4d8e010c-00b8-11ee-3807-13e4fb91a23a
using DrWatson

# ╔═╡ 334ddb4d-fb3f-4c5b-915d-a5d6ebe7b472
@quickactivate

# ╔═╡ 700e55b7-44b7-42b7-8f15-d6e7fcfc22c1
using GPLab

# ╔═╡ 569778ac-5f7d-4083-9597-a0996e147ee9
using BenchmarkTools

# ╔═╡ 4b921fa5-f8d1-4caa-b760-0ccc664118dd
using DataFrames

# ╔═╡ 548b86ca-26e6-43a3-86f6-649a335194e8
using RDatasets

# ╔═╡ c72b8adb-ad8c-409e-85c3-71a77fb62a36
mcycle = dataset("MASS", "mcycle")

# ╔═╡ 8e304cea-b1d2-443c-a3d0-1a8d49c3ff92
x = Matrix(mcycle[:, [:Times]])

# ╔═╡ b484c64b-393a-4530-b569-3fe044f84e9a
y = mcycle[:, :Accel]

# ╔═╡ 527a949d-ddc6-4397-8253-746a6b037c78
fitstd = fit(GPRegressor(), x, y)

# ╔═╡ 1006b2c5-38ec-4e90-8263-96abd03dad0c
fitstd.lml

# ╔═╡ d6eada9b-68e4-48df-b0a2-6f54989387d2
fitstd.params

# ╔═╡ ce0245a3-85bb-494b-b29d-580ecf0fda51
std_trial = @benchmark fit(GPRegressor(), x, y)

# ╔═╡ 366b18c5-f568-4085-8204-a4cfc5bf4c7a
begin
	mcycle_rep = combine(groupby(mcycle, :Times)) do sdf
    	(Accel=(sdf[:, :Accel],),)
	end
	mcycle_rep[!, :Accel] = first.(mcycle_rep[:, :Accel])
	mcycle_rep
end

# ╔═╡ a0915496-141d-440e-a8a3-ddfa290f0339
xrep = Matrix(mcycle_rep[:, [:Times]])

# ╔═╡ 09b044f5-ce80-4848-81f4-0d649d3e8805
yrep = mcycle_rep[:, :Accel]

# ╔═╡ 22edceaa-d07f-425b-b0a8-881b54bf2a67
fitrep = fit(RepGPRegressor(), xrep, yrep)

# ╔═╡ 02ca760f-5cd3-4cf7-acc8-f30aee810663
fitrep.lml

# ╔═╡ de92ca72-3de2-49a7-b848-7545b77c3f15
fitrep.params

# ╔═╡ 1d59d090-2e73-48a7-8d3e-1e6b892bbb4a
rep_trial = @benchmark fit(RepGPRegressor(), xrep, yrep)

# ╔═╡ 90c665ab-3122-4e3a-8645-b0a14ae013b5
ratio(median(std_trial), median(rep_trial))

# ╔═╡ bba0a0b8-a2fd-43f8-8945-19c3224edeac
lgpr = LaplaceGPRegressor(GaussianLikelihood())

# ╔═╡ eb36b1f9-c2ee-4596-8108-32f433c1aa01
@timev fit(lgpr, x, y)

# ╔═╡ 2c2dc8a7-bef6-49c1-9e24-d140dca7d0d4
@timev fit(lgpr, x, y)

# ╔═╡ 0f38804d-a6b7-40cb-988c-fd684b336c74
lrgpr = LaplaceGPRegressor(Replicate(GaussianLikelihood()))

# ╔═╡ 695c10cd-eb50-45a8-8f6e-78b27b092c9e
@timev fit(lrgpr, xrep, yrep)

# ╔═╡ df2f99d9-6b83-41e3-aede-457045addc73
@timev fit(lrgpr, xrep, yrep)

# ╔═╡ Cell order:
# ╠═4d8e010c-00b8-11ee-3807-13e4fb91a23a
# ╠═334ddb4d-fb3f-4c5b-915d-a5d6ebe7b472
# ╠═700e55b7-44b7-42b7-8f15-d6e7fcfc22c1
# ╠═569778ac-5f7d-4083-9597-a0996e147ee9
# ╠═4b921fa5-f8d1-4caa-b760-0ccc664118dd
# ╠═548b86ca-26e6-43a3-86f6-649a335194e8
# ╠═c72b8adb-ad8c-409e-85c3-71a77fb62a36
# ╠═8e304cea-b1d2-443c-a3d0-1a8d49c3ff92
# ╠═b484c64b-393a-4530-b569-3fe044f84e9a
# ╠═527a949d-ddc6-4397-8253-746a6b037c78
# ╠═1006b2c5-38ec-4e90-8263-96abd03dad0c
# ╠═d6eada9b-68e4-48df-b0a2-6f54989387d2
# ╠═ce0245a3-85bb-494b-b29d-580ecf0fda51
# ╠═366b18c5-f568-4085-8204-a4cfc5bf4c7a
# ╠═a0915496-141d-440e-a8a3-ddfa290f0339
# ╠═09b044f5-ce80-4848-81f4-0d649d3e8805
# ╠═22edceaa-d07f-425b-b0a8-881b54bf2a67
# ╠═02ca760f-5cd3-4cf7-acc8-f30aee810663
# ╠═de92ca72-3de2-49a7-b848-7545b77c3f15
# ╠═1d59d090-2e73-48a7-8d3e-1e6b892bbb4a
# ╠═90c665ab-3122-4e3a-8645-b0a14ae013b5
# ╠═bba0a0b8-a2fd-43f8-8945-19c3224edeac
# ╠═eb36b1f9-c2ee-4596-8108-32f433c1aa01
# ╠═2c2dc8a7-bef6-49c1-9e24-d140dca7d0d4
# ╠═0f38804d-a6b7-40cb-988c-fd684b336c74
# ╠═695c10cd-eb50-45a8-8f6e-78b27b092c9e
# ╠═df2f99d9-6b83-41e3-aede-457045addc73
