abstract type RepLikelihood <: Likelihood end

nparam(lik::RepLikelihood) = nparam(base_likelihood(lik))

fisher_info(lik::RepLikelihood, θ, y) = fisher_info(base_likelihood(lik), θ, nrepl(lik, y))

postpred(lik::RepLikelihood, θ) = postpred(base_likelihood(lik), θ)

struct RepGaussian <: RepLikelihood end

base_likelihood(::RepGaussian) = Gaussian()

function compute_statistics(::RepGaussian, y)
    u = mean.(y)
    v = varm.(y, u; corrected=false)
    return (u, v, m=length.(y))
end

nrepl(::RepGaussian, y) = y.m

function init_latent(::RepGaussian, y)
    n = sum(y.m)
    mean = sum(y.u .* y.m) / n
    var = sum(y.v .* y.m) / n
    len = length(y.m)
    return [fill(mean, len) ; fill(log(var), len)]
end

function loglik(lik::RepGaussian, θ, y)
    mean, logvar = _each_param(lik, θ)
    prec = exp.(-logvar)
    return -0.5 * sum(y.m .* (
        log2π .+ 
        logvar .+
        prec .* ((y.u .- mean).^2 .+ y.v)
    ))
end

function grad_loglik(lik::RepGaussian, θ, y)
    mean, logvar = _each_param(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    dmean = y.m .* prec .* z
    dlogvar = 0.5 .* y.m .* (-1 .+ (prec .* (z.^2 .+ y.v)))
    return [dmean ; dlogvar]
end

function hess_loglik(lik::RepGaussian, θ, y)
    mean, logvar = _each_param(lik, θ)
    prec = exp.(-logvar)
    z = y.u .- mean
    ddm = Diagonal(-y.m .* prec)
    dm_dlv = Diagonal(-y.m .* z .* prec)
    ddlv = Diagonal(-0.5 .* y.m .* prec .* (z.^2 .+ y.v))
    return [ddm dm_dlv ; dm_dlv ddlv]
end

