abstract type Likelihood end

_params(lik::Likelihood, θ) = eachrow(reshape(θ, :, nparam(lik)))

function loglik(lik::Likelihood, θ, y)
    return sum(loglik1.(Ref(lik), _params(lik, θ), y))
end

function grad_loglik(lik::Likelihood, θ, y)
    return vec(stack(grad_loglik1.(Ref(lik), _params(lik, θ), y))')
end

function _diag_blocks(a::AbstractVector{T}) where {T <: AbstractMatrix}
    return hvcat(size(a[1], 1), Diagonal.(eachcol(reshape(stack(a; dims=1), length(a), length(a[1]))))...)'
end

function _diag_blocks(a::AbstractVector{T}) where {T <: Diagonal}
    return Diagonal(vec(stack(diag.(a))'))
end

function hess_loglik(lik::Likelihood, θ, y)
    return _diag_blocks(hess_loglik1.(Ref(lik), _params(lik, θ), y))
end

function fisher_info(lik::Likelihood, θ, y)
    return _diag_blocks(fisher_info1.(Ref(lik), _params(lik, θ) .* nobs(lik, y)))
end

abstract type SimpleLikelihood <: Likelihood end

sufficient_stats(::SimpleLikelihood, y) = y

nobs(::SimpleLikelihood, y) = ones(size(y))

struct Replicate{Tb <: SimpleLikelihood} <: Likelihood
    base::Tb
end

sufficient_stats(::Replicate{Tb}, y) where {Tb} = y

nobs(::Replicate{Tb}, y) where {Tb} = length.(y)

nparam(lik::Replicate{Tb}) where {Tb} = nparam(lik.base)

function init_latent(lik::Replicate{Tb}, y) where {Tb}
    return init_latent(lik.base, reduce(vcat, y))
end

function loglik1(lik::Replicate{Tb}, θ, y) where {Tb}
    return sum(loglik1.(Ref(lik.base), Ref(θ), y))
end

function grad_loglik1(lik::Replicate{Tb}, θ, y) where {Tb}
    return sum(grad_loglik1.(Ref(lik.base), Ref(θ), y))
end

function hess_loglik1(lik::Replicate{Tb}, θ, y) where {Tb}
    return sum(hess_loglik1.(Ref(lik.base), Ref(θ), y))
end

function fisher_info1(lik::Replicate{Tb}, θ) where {Tb}
    return fisher_info1(lik.base, θ)
end

function postpred(lik::Replicate{Tb}, θ) where {Tb}
    return postpred(lik.base, θ)
end
