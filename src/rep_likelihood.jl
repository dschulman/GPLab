abstract type RepLikelihood <: Likelihood end

nparam(lik::RepLikelihood) = nparam(base_likelihood(lik))

fisher_info(lik::RepLikelihood, θ, y) = fisher_info(base_likelihood(lik), θ, nrepl(lik, y))

postpred(lik::RepLikelihood, θ) = postpred(base_likelihood(lik), θ)

struct RepGaussianLikelihood <: RepLikelihood end

base_likelihood(::RepGaussianLikelihood) = GaussianLikelihood()

function compute_statistics(::RepGaussianLikelihood, y)
    u = mean.(y)
    v = varm.(y, u; corrected=false)
    return (u, v, m=length.(y))
end

nrepl(::RepGaussianLikelihood, y) = y.m

function init_latent(::RepGaussianLikelihood, y)
    n = sum(y.m)
    mean = sum(y.u .* y.m) / n
    var = sum(y.v .* y.m) / n
    len = length(y.m)
    return [fill(mean, len) ; fill(log(var), len)]
end

function loglik(lik::RepGaussianLikelihood, θ, y)
    mean, logvar = _each_param(lik, θ)
    prec = exp.(-logvar)
    return -0.5 * sum(y.m .* (
        log2π .+ 
        logvar .+
        prec .* ((y.u .- mean).^2 .+ y.v)
    ))
end

function grad_loglik(lik::RepGaussianLikelihood, θ, y)
    mean, logvar = _each_param(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    dmean = y.m .* prec .* z
    dlogvar = 0.5 .* y.m .* (-1 .+ (prec .* (z.^2 .+ y.v)))
    return [dmean ; dlogvar]
end

function hess_loglik(lik::RepGaussianLikelihood, θ, y)
    mean, logvar = _each_param(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    ddm = Diagonal(-y.m .* prec)
    dm_dlv = Diagonal(-y.m .* z .* prec)
    ddlv = Diagonal(-0.5 .* y.m .* prec .* (z.^2 .+ y.v))
    return [ddm dm_dlv ; dm_dlv ddlv]
end

