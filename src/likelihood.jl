abstract type Likelihood end

_params(lik::Likelihood, θ) = eachcol(reshape(θ, :, nparam(lik)))

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

compute_stats(::SimpleLikelihood, y) = y

struct Replicate{Tb <: SimpleLikelihood} <: Likelihood
    base::Tb
end

nparam(lik::Replicate) = nparam(lik.base)

function compute_stats(lik::Replicate, y)
    m = length.(y)
    M = Diagonal(repeat(m, nparam(lik)))
    u = ones.(m)
    U = cat(repeat(u, nparam(lik))...; dims=(1,2))
    return (y=reduce(vcat, y), U, M)
end

init_latent(lik::Replicate, y) = init_latent(lik.base, y.y)

loglik(lik::Replicate, θ, y) = loglik(lik.base, y.U * θ, y.y)

grad_loglik(lik::Replicate, θ, y) = y.U' * grad_loglik(lik.base, y.U * θ, y.y)

hess_loglik(lik::Replicate, θ, y) = y.U' * hess_loglik(lik.base, y.U * θ, y.y) * y.U

fisher_info(lik::Replicate, θ, y) = y.M * fisher_info(lik.base, θ, nothing)

postpred(lik::Replicate, θ) = postpred(lik.base, θ)
