struct GammaLikelihood <: Likelihood end

nparam(::GammaLikelihood) = 2

function init_latent(::GammaLikelihood, y, w)
    # convert() is needed because Distributions.suffstats is overspecialized
    dist = fit_mle(Gamma, y, convert(Vector{Float64}, w))
    return [log(mean(dist)), log(shape(dist))]
end

lognormalizer(::GammaLikelihood) = 0

function loglik1(::GammaLikelihood, (logm, logk), y)
    m = exp(logm)
    k = exp(logk)
    return -loggamma(k) - k*(logm - logk) + xlogy(k-1, y) - (y * k / m)
end

function grad_loglik1(::GammaLikelihood, (logm, logk), y)
    m = exp(logm)
    k = exp(logk)
    dlogm = k * (y - m) / m
    dlogk = k * (-digamma(k) - logm + logk + 1 + log(y) - (y / m))
    return [dlogm, dlogk]
end

function hess_loglik1(::GammaLikelihood, (logm, logk), y)
    m = exp(logm)
    k = exp(logk)
    ddlm = - k * y / m
    ddlk = k * (-digamma(k) - k*trigamma(k) - logm + logk + 1 + log(y) - (y / m))
    dlm_dlk = k * (y - m) / m
    return [ ddlm dlm_dlk ; dlm_dlk ddlk ]
end

function fisher_info1(::GammaLikelihood, (_, logk))
    k = exp(logk)
    return Diagonal([k, k*k*trigamma(k)])
end

# TODO is it worth caching quadrature weights?
function _approx_eψk(elogk, vlogk; npoints=7)
    x, w = gausshermite(npoints)
    f = digamma.(exp.((x .* sqrt(2vlogk)) .+ elogk))
    return (w ⋅ f) / sqrtπ
end

# TODO allow choosing # of quadrature points?
function postpred(::GammaLikelihood, lmean, lvar)
    elogm, elogk = lmean
    vlogm, vlogk = diag(lvar)
    ey = exp(elogm + (vlogm / 2))
    elogy = _approx_eψk(elogk, vlogk) + elogm - elogk
    return fit_mle(Gamma, Distributions.GammaStats(ey, elogy, 1.0))
end
