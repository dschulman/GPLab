struct TLikelihood{Tdf <: Real} <: Likelihood
    df::Tdf
end

nparam(::TLikelihood) = 2

function init_latent(::TLikelihood, y, w)
    m, v = mean_and_var(y, w; corrected=false)
    return [m, log(v)]
end

function lognormalizer(lik::TLikelihood)
    return (
        loggamma((lik.df + 1) / 2) - 
        loggamma(lik.df / 2) - 
        0.5 * (logπ + log(lik.df))
    )
end

function loglik1(lik::TLikelihood, (m, logv), y)
    z2invv = (y - m)^2 * exp(-logv)
    return -0.5 * (logv + xlog1py(lik.df + 1, z2invv / lik.df))
end

function grad_loglik1(lik::TLikelihood, (m, logv), y)
    z = y - m
    z2 = z^2
    vdf = exp(logv) * lik.df
    dm = (lik.df + 1) * z / (z2 + vdf)
    dlogv = -0.5 + (0.5 * (lik.df + 1) * z2 / (z2 + vdf))
    return [dm, dlogv]
end

function hess_loglik1(lik::TLikelihood, (m, logv), y)
    z = y - m
    z2 = z ^ 2
    vdf = exp(logv) * lik.df
    z2vdf2 = (z2 + vdf)^2
    ddm = (lik.df + 1) * (z2 - vdf) / z2vdf2
    ddlv = -0.5 * (lik.df + 1) * z2 * vdf / z2vdf2
    dm_dlv = -(lik.df + 1) * z * vdf / z2vdf2
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info1(lik::TLikelihood, (_, logv))
    im = (lik.df + 1) * exp(-logv) / (lik.df + 3)
    ilv = 0.5 * lik.df / (lik.df + 3)
    return Diagonal([im, ilv])
end

function postpred(lik::TLikelihood, lmean, lvar)
    em, elogv = lmean
    vm, vlogv = diag(lvar)
    yvar = vm + (lik.df / (lik.df - 2) * exp(elogv + (vlogv / 2)))
    return em, yvar
end
