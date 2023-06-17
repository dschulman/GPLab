struct TLikelihood{Tdf <: Real} <: SimpleLikelihood
    df::Tdf
    logZ::Float64

    # Check df is positive
    # Precompute the (rather expensive) normalization constant
    function TLikelihood(df::Tdf) where {Tdf <: Real}
        if df ≤ 0
            error("Student t df must be positive")
        end
        logZ = (
            loggamma((df + 1) / 2) - 
            loggamma(df / 2) - 
            0.5 * (logπ + log(df))
        )
        return new{Tdf}(df, logZ)
    end
end

nparam(::TLikelihood) = 2

function init_latent(::TLikelihood, y)
    m, v = mean_and_var(y; corrected=false)
    return [m, log(v)]
end

function loglik(lik::TLikelihood, θ, y)
    n = length(y)
    df = lik.df
    m, logv = _params(lik, θ)
    z2invv = (y .- m).^2 .* exp.(-logv)
    return (n * lik.logZ) - (0.5 * sum(logv .+ xlog1py.(df + 1, z2invv ./ df)))
end

function grad_loglik(lik::TLikelihood, θ, y)
    m, logv = _params(lik, θ)
    z = y .- m
    z2 = z .^ 2
    vdf = exp.(logv) .* lik.df
    dm = (lik.df + 1) .* z ./ (z2 .+ vdf)
    dlogv = -0.5 .+ (0.5 .* (lik.df + 1) .* z2 ./ (z2 .+ vdf))
    return [dm ; dlogv]
end

function hess_loglik(lik::TLikelihood, θ, y)
    m, logv = _params(lik, θ)
    z = y .- m
    z2 = z .^ 2
    vdf = exp.(logv) .* lik.df
    z2vdf2 = (z2 .+ vdf).^2
    ddm = Diagonal((lik.df + 1) .* (z2 .- vdf) ./ z2vdf2)
    ddlv = Diagonal(-0.5 * (lik.df + 1) .* z2 .* vdf ./ z2vdf2)
    dm_dlv = Diagonal(-(lik.df + 1) .* z .* vdf ./ z2vdf2)
    return [ddm dm_dlv ; dm_dlv ddlv]
end

function fisher_info(lik::TLikelihood, θ, _)
    _, logv = _params(lik, θ)
    im = (lik.df + 1) .* exp.(-logv) ./ (lik.df + 3)
    ilv = fill(0.5 * lik.df / (lik.df + 3), length(logv))
    return Diagonal([im ; ilv])
end

function postpred(lik::TLikelihood, θ)
    em, elogv = mean(θ)
    vm, vlogv = var(θ)
    yvar = vm + (lik.df / (lik.df - 2) * exp(elogv + (vlogv / 2)))
    return Normal(em, sqrt(yvar))
end
