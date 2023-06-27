struct GaussianLikelihood <: Likelihood end

nparam(lik::GaussianLikelihood) = 2

function init_latent(::GaussianLikelihood, y, w)
    m, v = mean_and_var(y, Weights(w); corrected=false)
    return [m, log(v)]
end

lognormalizer(::GaussianLikelihood) = - log2π / 2

function loglik1(::GaussianLikelihood, (mean, logvar), y)
    return -0.5 * (logvar + ((y-mean)^2 * exp(-logvar)))
end

function grad_loglik1(::GaussianLikelihood, (mean, logvar), y)
    z = y - mean
    prec = exp(-logvar)
    dmean = z * prec
    dlogvar = -0.5 + (0.5 * z * z * prec)
    return [dmean, dlogvar]
end

function hess_loglik1(::GaussianLikelihood, (mean, logvar), y)
    z = y - mean
    prec = exp(-logvar)
    ddm = -prec
    dm_dlv = -z * prec
    ddlv = -0.5 * z * z * prec
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info1(::GaussianLikelihood, (_, logvar))
    return Diagonal([exp(-logvar), 0.5])
end

function postpred(::GaussianLikelihood, θ)
    emean, elogvar = mean(θ)
    vmean, vlogvar = var(θ)
    yvar = vmean + exp(elogvar + (vlogvar / 2))
    return Normal(emean, sqrt(yvar))
end

struct ReplicateGaussianStats{T}
    u::T
    v::T
end

function compute_stats(::Replicate{GaussianLikelihood}, y)
    u = mean.(y)
    v = varm.(y, u; corrected=false)
    return StructArray{ReplicateGaussianStats}((u=u, v=v))
end

function init_latent(::Replicate{GaussianLikelihood}, y, w)
    n = sum(w)
    u = sum(y.u .* w) / n
    v = sum(y.v .* w) / n
    return [u, log(v)]
end

function loglik1(::Replicate{GaussianLikelihood}, (mean, logvar), y)
    prec = exp(-logvar)
    z = y.u - mean
    return -0.5 * (logvar + (prec * (z^2 + y.v)))
end

function grad_loglik1(::Replicate{GaussianLikelihood}, (mean, logvar), y)
    prec = exp(-logvar)
    z = y.u - mean
    dmean = prec * z
    dlogvar = 0.5 * (-1 + (prec * (z^2 + y.v)))
    return [dmean, dlogvar]
end

function hess_loglik1(::Replicate{GaussianLikelihood}, (mean, logvar), y)
    prec = exp(-logvar)
    z = y.u - mean
    ddm = - prec
    dm_dlv = - z * prec
    ddlv = -0.5 * prec * (z^2 + y.v)
    return [ddm dm_dlv ; dm_dlv ddlv]
end
