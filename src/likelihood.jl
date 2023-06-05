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

struct GaussianLikelihood <: SimpleLikelihood end

nparam(lik::GaussianLikelihood) = 2

function init_latent(::GaussianLikelihood, y)
    m, v = mean_and_var(y; corrected=false)
    return [m, log(v)]
end

function loglik1(::GaussianLikelihood, θ, y)
    mean, logvar = θ
    return -0.5 * (log2π + logvar + ((y-mean)^2 * exp(-logvar)))
end

function grad_loglik1(::GaussianLikelihood, θ, y)
    mean, logvar = θ
    z = y - mean
    prec = exp(-logvar)
    dmean = z * prec
    dlogvar = -0.5 + (0.5 * z * z * prec)
    return [dmean, dlogvar]
end

function hess_loglik1(lik::GaussianLikelihood, θ, y)
    mean, logvar = θ
    z = y - mean
    prec = exp(-logvar)
    ddm = -prec
    dm_dlv = -z * prec
    ddlv = -0.5 * z * z * prec
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info1(::GaussianLikelihood, θ)
    logvar = θ[2]
    return Diagonal([exp(-logvar), 0.5])
end

function postpred(::GaussianLikelihood, θ)
    emean, elogvar = mean(θ)
    vmean, vlogvar = var(θ)
    yvar = vmean + exp(elogvar + (vlogvar / 2))
    return Normal(emean, sqrt(yvar))
end
