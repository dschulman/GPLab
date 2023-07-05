struct GammaLikelihood <: Likelihood end

nparam(::GammaLikelihood) = 2

function init_latent(::GammaLikelihood, y, w)
    # convert() is needed because Distributions.suffstats is overspecialized
    dist = fit_mle(Gamma, y, convert(Vector{Float64}, w))
    return [log(mean(dist)), -log(shape(dist))]
end

lognormalizer(::GammaLikelihood) = 0

function loglik1(::GammaLikelihood, (logm, logd), y)
    invm = exp(-logm)
    invd = exp(-logd)
    return -loggamma(invd) - invd*(logm + logd) + xlogy(invd-1, y) - (y * invm * invd)
end

function grad_loglik1(::GammaLikelihood, (logm, logd), y)
    invm = exp(-logm)
    invd = exp(-logd)
    dlogm = invd * ((y * invm) - 1)
    dlogd = invd * (digamma(invd) - 1 - log(y) + logm + logd + (y * invm))
    return [dlogm, dlogd]
end

function hess_loglik1(::GammaLikelihood, (logm, logd), y)
    invm = exp(-logm)
    invd = exp(-logd)
    ddlm = - (y * invm * invd)
    ddld = invd * (-invd*trigamma(invd) - digamma(invd) + 2 + log(y) - logm - logd - (y * invm))
    dlm_dld = invd * (1 - (y * invm))
    return [ ddlm dlm_dld ; dlm_dld ddld]
end

function fisher_info1(::GammaLikelihood, (_, logd))
    invd = exp(-logd)
    return Diagonal([invd, invd*invd*trigamma(invd) - invd])
end

# TODO is it worth caching quadrature weights?
function _approx_eψk(elogd, vlogd; npoints=7)
    x, w = gausshermite(npoints)
    f = digamma.(exp.(-(x .* sqrt(2vlogd)) .- elogd))
    return (w ⋅ f) / sqrtπ
end

# TODO allow choosing # of quadrature points?
function postpred(::GammaLikelihood, lmean, lvar)
    elogm, elogd = lmean
    vlogm, vlogd = diag(lvar)
    ey = exp(elogm + (vlogm / 2))
    elogy = _approx_eψk(elogd, vlogd) + elogm + elogd
    return fit_mle(Gamma, Distributions.GammaStats(ey, elogy, 1.0))
end

struct ReplicateGammaStats{T}
    u::T
    v::T
end

function compute_stats(::Replicate{GammaLikelihood}, y)
    u = mean.(y)
    v = [mean(log.(yi)) for yi in y]
    return StructArray{ReplicateGammaStats}((u=u, v=v))
end

function init_latent(::Replicate{GammaLikelihood}, y, w)
    u = sum(y.u, w)
    v = sum(y.v, w)
    dist = fit_mle(Gamma, Distributions.GammaStats(u, v, sum(w)))
    return [log(mean(dist)), -log(shape(dist))]
end

function loglik1(::Replicate{GammaLikelihood}, (logm, logd), y)
    invm = exp(-logm)
    invd = exp(-logd)
    return -loggamma(invd) - invd*(logm + logd) + (invd - 1)*y.v - (y.u * invm * invd)
end

function grad_loglik1(::Replicate{GammaLikelihood}, (logm, logd), y)
    invm = exp(-logm)
    invd = exp(-logd)
    dlogm = invd * ((y.u * invm) - 1)
    dlogd = invd * (digamma(invd) - 1 - y.v + logm + logd + (y.u * invm))
    return [dlogm, dlogd]
end

function hess_loglik1(::Replicate{GammaLikelihood}, (logm, logd), y)
    invm = exp(-logm)
    invd = exp(-logd)
    ddlm = - (y.u * invm * invd)
    ddld = invd * (-invd*trigamma(invd) - digamma(invd) + 2 + y.v - logm - logd - (y.u * invm))
    dlm_dld = invd * (1 - (y.u * invm))
    return [ ddlm dlm_dld ; dlm_dld ddld]
end
