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

function hess_loglik1(::GaussianLikelihood, θ, y)
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
