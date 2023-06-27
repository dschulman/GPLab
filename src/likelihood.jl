abstract type Likelihood end

_params(lik::Likelihood, θ) = eachrow(reshape(θ, :, nparam(lik)))

compute_stats(::Likelihood, y) = y

default_weight(::Likelihood, y) = 1

function loglik(lik::Likelihood, θ, y, w)
    ln = lognormalizer(lik)
    return wsum(ln .+ loglik1.(Ref(lik), _params(lik, θ), y), w)
end

function _block_vec(a::AbstractVector{T}) where {T <: AbstractVector}
    return vec(stack(a)')
end

function _block_vec(a::AbstractVector{T}) where {T <: Number}
    return a
end

function grad_loglik(lik::Likelihood, θ, y, w)
    return _block_vec(w .* grad_loglik1.(Ref(lik), _params(lik, θ), y))
end

function _diag_blocks(a::AbstractVector{T}) where {T <: AbstractMatrix}
    return hvcat(size(a[1], 1), Diagonal.(eachcol(reshape(stack(a; dims=1), length(a), length(a[1]))))...)'
end

function _diag_blocks(a::AbstractVector{T}) where {T <: Diagonal}
    return Diagonal(vec(stack(diag.(a))'))
end

function _diag_blocks(a::AbstractVector{T}) where {T <: Number}
    return Diagonal(a)
end

function hess_loglik(lik::Likelihood, θ, y, w)
    return _diag_blocks(w .* hess_loglik1.(Ref(lik), _params(lik, θ), y))
end

function fisher_info(lik::Likelihood, θ, w)
    return _diag_blocks(w .* fisher_info1.(Ref(lik), _params(lik, θ)))
end

struct Replicate{Tb <: Likelihood} <: Likelihood
    base::Tb
end

nparam(lik::Replicate) = nparam(lik.base)

default_weight(::Replicate, y) = length(y)

function init_latent(lik::Replicate, y, w)
    yy = reduce(vcat, y)
    m = length.(y)
    ww = reduce(vcat, fill.(w ./ m, m))
    return init_latent(lik.base, yy, ww)
end

lognormalizer(lik::Replicate) = lognormalizer(lik.base)

loglik1(lik::Replicate, θ, y) = mean(loglik1.(Ref(lik.base), Ref(θ), y))

grad_loglik1(lik::Replicate, θ, y) = mean(grad_loglik1.(Ref(lik.base), Ref(θ), y))

hess_loglik1(lik::Replicate, θ, y) = mean(hess_loglik1.(Ref(lik.base), Ref(θ), y))

fisher_info1(lik::Replicate, θ) = fisher_info1(lik.base, θ)

postpred(lik::Replicate, θ) = postpred(lik.base, θ)
