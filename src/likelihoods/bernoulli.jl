struct BernoulliLogitLikelihood <: Likelihood end

nparam(::BernoulliLogitLikelihood) = 1

init_latent(::BernoulliLogitLikelihood, y, w) = [logit(mean(y, w))]

lognormalizer(::BernoulliLogitLikelihood) = 1

loglik1(::BernoulliLogitLikelihood, (θ,), y) = -log1pexp((1 - 2y) * θ)

grad_loglik1(::BernoulliLogitLikelihood, (θ,), y) = y - logistic(θ)

function hess_loglik1(::BernoulliLogitLikelihood, (θ,), _)
    p = logistic(θ)
    return - p * (1 - p)
end

function fisher_info1(::BernoulliLogitLikelihood, (θ,))
    p = logistic(θ)
    return p * (1 - p)
end

function postpred(::BernoulliLogitLikelihood, lmean, lvar)
    # The logistic-normal integral is intractable:
    # ∫ logistic(x) * N(x | μ, σ2) dx
    #
    # Method from Williams and Barber (1998), "Bayesian Classification with Gaussian Processes".
    # Approximate the logistic function with a linear combination of normcdfs.
    # logistic(x) ≈ c ⋅ normcdf.(λ .* x)
    #
    # The result is a linear combination of normcdfs (via standard formula):
    # ∫ normcdf(λ * x) * N(x | μ, σ²) = normcdf(μ * λ / sqrt(1 + (σ2 * λ²)))
    #
    # The paper gives no detail on how the basis functions are chosen, but
    # from a bit of exploration, they are an approximate (local?) minimum in square
    # error for logistic(x) at the points x.
    # I'm not clear how x is chosen.
    # Coefficients are chosen by minimizing square error given λ.
    λ = [0.41, 0.4, 0.37, 0.44, 0.39] .* sqrt2
    # x = [0, 0.6, 2, 3.5, 4.5, Inf]
    # b = logistic.(x)
    # A = normcdf.(x * λ')
    # c = A \ b
    c = [-1854.8214151380894, 3516.8989365473444, 221.29346712948822, 128.12323805570333, -2010.4942265944464]
    z = only(lmean) .* λ ./ sqrt.(1 .+ (only(lvar) .* λ.^2))
    p = c ⋅ normcdf.(z)
    return p, p*(1-p)
end

struct BernoulliProbitLikelihood <: Likelihood end

nparam(::BernoulliProbitLikelihood) = 1

init_latent(::BernoulliProbitLikelihood, y, w) = [norminvcdf(mean(y, w))]

lognormalizer(::BernoulliProbitLikelihood) = 1

loglik1(::BernoulliProbitLikelihood, (θ,), y) = normlogcdf((2y - 1) * θ)

function grad_loglik1(::BernoulliProbitLikelihood, (θ,), y)
    t = 2y - 1
    tθ = t * θ
    return t * normpdf(tθ) / normcdf(tθ)
end

function hess_loglik1(lik::BernoulliProbitLikelihood, (θ,), y)
    t = 2y - 1
    tθ = t * θ
    g = t * normpdf(tθ) / normcdf(tθ)
    return -g * (θ + g)
end

function fisher_info1(::BernoulliProbitLikelihood, (θ,))
    p = normcdf(θ)
    return normpdf(θ)^2 / p / (1 - p)
end

function postpred(::BernoulliProbitLikelihood, lmean, lvar)
    p = normcdf(only(lmean) / sqrt(1 + only(lvar)))
    return p, p*(1-p)
end
