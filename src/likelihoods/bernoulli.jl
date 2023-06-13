struct BernoulliLogitLikelihood <: SimpleLikelihood end

nparam(::BernoulliLogitLikelihood) = 1

init_latent(::BernoulliLogitLikelihood, y) = [logit(mean(y))]

function loglik(::BernoulliLogitLikelihood, θ, y)
    return sum(-log1pexp.((1 .- 2y) .* θ))
end

function grad_loglik(::BernoulliLogitLikelihood, θ, y)
    return y .- logistic.(θ)
end

function hess_loglik(::BernoulliLogitLikelihood, θ, _)
    p = logistic.(θ)
    return Diagonal(- p .* (1 .- p))
end

function fisher_info(::BernoulliLogitLikelihood, θ, _)
    p = logistic.(θ)
    return Diagonal(p .* (1 .- p))
end

function postpred(::BernoulliLogitLikelihood, θ)
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
    f_mean = only(mean(θ))
    f_var = only(var(θ))
    z = f_mean .* λ ./ sqrt.(1 .+ (f_var .* λ.^2))
    p = c ⋅ normcdf.(z)
    return Bernoulli(p)
end

struct BernoulliProbitLikelihood <: SimpleLikelihood end

nparam(::BernoulliProbitLikelihood) = 1

init_latent(::BernoulliProbitLikelihood, y) = [norminvcdf(mean(y))]

function loglik(::BernoulliProbitLikelihood, θ, y)
    return sum(normlogcdf.((2y .- 1) .* θ))
end

function grad_loglik(::BernoulliProbitLikelihood, θ, y)
    t = 2y .- 1
    tθ = t .* θ
    return t .* normpdf.(tθ) ./ normcdf.(tθ)
end

function hess_loglik(lik::BernoulliProbitLikelihood, θ, y)
    g = grad_loglik(lik, θ, y)
    return Diagonal(- g .* (θ .+ g))
end

function fisher_info(::BernoulliProbitLikelihood, θ, _)
    p = normcdf.(θ)
    return Diagonal(normpdf.(θ).^2 ./ p ./ (1 .- p))
end

function postpred(::BernoulliProbitLikelihood, θ)
    f_mean = mean(θ)[1]
    f_var = var(θ)[1]
    return normcdf(f_mean / sqrt(1 + f_var))
end
