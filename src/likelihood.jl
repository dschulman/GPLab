abstract type Likelihood end

_params(lik::Likelihood, θ) = eachrow(reshape(θ, :, nparam(lik)))

compute_stats(::Likelihood, y) = y

function loglik(lik::Likelihood, θ, y)
    ln = sum(nobs(lik, y)) * lognormalizer(lik)
    return ln + sum(loglik1.(Ref(lik), _params(lik, θ), y))
end

function _block_vec(a::AbstractVector{T}) where {T <: AbstractVector}
    return vec(stack(a)')
end

function _block_vec(a::AbstractVector{T}) where {T <: Number}
    return a
end

function grad_loglik(lik::Likelihood, θ, y)
    return _block_vec(grad_loglik1.(Ref(lik), _params(lik, θ), y))
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

function hess_loglik(lik::Likelihood, θ, y)
    return _diag_blocks(hess_loglik1.(Ref(lik), _params(lik, θ), y))
end

function fisher_info(lik::Likelihood, θ, y)
    return _diag_blocks(fisher_info1.(Ref(lik), _params(lik, θ)) .* nobs(lik, y))
end

abstract type SimpleLikelihood <: Likelihood end

nobs(::SimpleLikelihood, y) = ones(length(y))

struct Replicate{Tb <: SimpleLikelihood} <: Likelihood
    base::Tb
end

nparam(lik::Replicate) = nparam(lik.base)

nobs(::Replicate, y) = length.(y)

init_latent(lik::Replicate, y) = init_latent(lik.base, reduce(vcat, y))

lognormalizer(lik::Replicate) = lognormalizer(lik.base)

loglik1(lik::Replicate, θ, y) = sum(loglik1.(Ref(lik.base), Ref(θ), y))

grad_loglik1(lik::Replicate, θ, y) = sum(grad_loglik1.(Ref(lik.base), Ref(θ), y))

hess_loglik1(lik::Replicate, θ, y) = sum(hess_loglik1.(Ref(lik.base), Ref(θ), y))

fisher_info1(lik::Replicate, θ) = fisher_info1(lik.base, θ)

postpred(lik::Replicate, θ) = postpred(lik.base, θ)
