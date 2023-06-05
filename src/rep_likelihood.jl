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

function sufficient_stats(::Replicate{GaussianLikelihood}, y)
    u, v = mean_and_var(y; corrected=false)
    return (u, v, m=length(y))
end

function nobs(::Replicate{GaussianLikelihood}, y)
    return [yi.m for yi in y]
end

function init_latent(::Replicate{GaussianLikelihood}, y)
    n = sum([yi.m for yi in y])
    u = sum([yi.u * yi.m for yi in y]) / n
    v = sum([yi.v * yi.m for yi in y]) / n
    return [u, log(v)]
end

function loglik1(::Replicate{GaussianLikelihood}, θ, y)
    mean, logvar = θ
    prec = exp(-logvar)
    z = y.u - mean
    return -0.5 * y.m * (log2π + logvar + (prec * (z^2 + y.v)))
end

function grad_loglik1(::Replicate{GaussianLikelihood}, θ, y)
    mean, logvar = θ
    prec = exp(-logvar)
    z = y.u - mean
    dmean = y.m * prec * z
    dlogvar = 0.5 * y.m * (-1 + (prec * (z^2 + y.v)))
    return [dmean, dlogvar]
end

function hess_loglik1(::Replicate{GaussianLikelihood}, θ, y)
    mean, logvar = θ
    prec = exp(-logvar)
    z = y.u - mean
    ddm = -y.m * prec
    dm_dlv = -y.m * z * prec
    ddlv = -0.5 * y.m * prec * (z^2 + y.v)
    return [ddm dm_dlv ; dm_dlv ddlv]
end
