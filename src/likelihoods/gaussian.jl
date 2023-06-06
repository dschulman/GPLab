struct GaussianLikelihood <: SimpleLikelihood end

nparam(lik::GaussianLikelihood) = 2

function init_latent(::GaussianLikelihood, y)
    m, v = mean_and_var(y; corrected=false)
    return [m, log(v)]
end

function loglik(lik::GaussianLikelihood, θ, y)
    mean, logvar = _params(lik, θ)
    return -0.5 * sum(@. log2π + logvar + ((y-mean)^2 * exp(-logvar)))
end

function grad_loglik(lik::GaussianLikelihood, θ, y)
    mean, logvar = _params(lik, θ)
    z = y .- mean
    prec = exp.(-logvar)
    dmean = z .* prec
    dlogvar = -0.5 .+ (0.5 .* z .* z .* prec)
    return [dmean ; dlogvar]
end

function hess_loglik(lik::GaussianLikelihood, θ, y)
    mean, logvar = _params(lik, θ)
    z = y .- mean
    prec = exp.(-logvar)
    ddm = Diagonal(-prec)
    dm_dlv = Diagonal(-z .* prec)
    ddlv = Diagonal(-0.5 .* z .* z .* prec)
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info(lik::GaussianLikelihood, θ, _)
    _, logvar = _params(lik, θ)
    return Diagonal([exp.(-logvar) ; fill(0.5, length(logvar))])
end

function postpred(::GaussianLikelihood, θ)
    emean, elogvar = mean(θ)
    vmean, vlogvar = var(θ)
    yvar = vmean + exp(elogvar + (vlogvar / 2))
    return Normal(emean, sqrt(yvar))
end

function compute_stats(::Replicate{GaussianLikelihood}, y)
    m = length.(y)
    u = mean.(y)
    v = varm.(y, u; corrected=false)
    return (u, v, m, M=Diagonal(repeat(m, 2)))
end

function init_latent(::Replicate{GaussianLikelihood}, y)
    n = sum(y.m)
    u = sum(y.u .* y.m) / n
    v = sum(y.v .* y.m) / n
    return [u, log(v)]
end

function loglik(lik::Replicate{GaussianLikelihood}, θ, y)
    mean, logvar = _params(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    return -0.5 * sum(@. y.m * (log2π + logvar + (prec * (z^2 + y.v))))
end

function grad_loglik(lik::Replicate{GaussianLikelihood}, θ, y)
    mean, logvar = _params(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    dmean = @. y.m * prec * z
    dlogvar = @. 0.5 * y.m * (-1 + (prec * (z^2 + y.v)))
    return [dmean ; dlogvar]
end

function hess_loglik(lik::Replicate{GaussianLikelihood}, θ, y)
    mean, logvar = _params(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    ddm = Diagonal(-y.m .* prec)
    dm_dlv = Diagonal(-y.m .* z .* prec)
    ddlv = Diagonal(@. -0.5 * y.m * prec * (z^2 + y.v))
    return [ddm dm_dlv ; dm_dlv ddlv]
end
