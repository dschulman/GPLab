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
    # Method from Williams and Barber (1998), "Bayesian Classification with Gaussian Processes".
    # Approximate the logistic function with a linear combination of 5 erfs.
    # This gives a closed form integral, which is also a linear combination of erfs.
    λ = [0.41, 0.4, 0.37, 0.44, 0.39]
    # x = [0, 0.6, 2, 3.5, 4.5, Inf]
    # b = logistic.(x)
    # A = (erf.(x * λ') .+ 1) ./ 2
    # c = A \ b
    c = [-1854.8214153838703, 3516.898937004956, 221.29346715559808, 128.12323807294905, -2010.4942268496336]
    f_mean = mean(θ)[1]
    f_var = var(θ)[1]
    alpha = inv(2 * f_var)
    gamma = f_mean .* λ
    integrals = 0.5 .* erf.(gamma .* sqrt.(alpha ./ (alpha .+ λ.^2)))
    p = (c ⋅ integrals) + (sum(c) / 2)
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
