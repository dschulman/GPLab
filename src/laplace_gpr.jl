mutable struct LaplaceGPRegressor{Tl<:Likelihood, Tk<:Kernel}
    lik::Tl
    kernel::Tk
end

LaplaceGPRegressor(lik::Tl) where {Tl<:Likelihood} = LaplaceGPRegressor(lik, Matern52Kernel())

struct LaplaceGPRegression{Tk<:AbstractVector{<:Kernel}}
    approx_lml
    params
    lik
    kernels::Tk
    x
    g
    HinvB
end

function init_params(gpr::LaplaceGPRegressor, x::AbstractMatrix)
    nlatent = nparam(gpr.lik)
    return (
        mean = zeros(nlatent),
        k = fill(_init_kernel_params(x), nlatent)
    )
end

function fit(gpr::LaplaceGPRegressor, x::AbstractMatrix, yraw::AbstractVector)
    y = compute_stats(gpr.lik, yraw)
    n = size(x, 1)
    initθ, unflatten = ParameterHandling.value_flatten(init_params(gpr, x))
    function _laplace_nlml_wrapper(θ)
        params = unflatten(θ)
        kernels = _kernel.(Ref(gpr.kernel), params.k)
        kms = kernelmatrix.(kernels, Ref(RowVecs(x)))
        K = cat(kms...; dims=(1,2))
        μ = vcat(fill.(params.mean, n)...)
        return _laplace_nlml(gpr.lik, n, K, μ, y)
    end
    res = Optim.optimize(
        _zygote_fg!(_laplace_nlml_wrapper),
        initθ,
        Optim.LBFGS(alphaguess=LineSearches.InitialStatic(scaled=true)),
        Optim.Options(show_trace=true, show_every=1)
    )
    approx_lml = - minimum(res)
    params = unflatten(Optim.minimizer(res))
    kernels = _kernel.(Ref(gpr.kernel), params.k)
    kms = kernelmatrix.(kernels, Ref(RowVecs(x)))
    K = cat(kms...; dims=(1,2))
    μ = vcat(fill.(params.mean, n)...)
    f, g = _posterior_mode(gpr.lik, n, K, μ, y)
    H = hess_loglik(gpr.lik, f, y)
    B = lu(I - (K * H))
    return LaplaceGPRegression(approx_lml, params, gpr.lik, kernels, x, g, H / B)
end

function _laplace_nlml(lik, n, K, μ, y)
    return _laplace_nlml_and_intermediates(lik, n, K, μ, y)[1]
end

function _laplace_nlml_and_intermediates(lik, n, K, μ, y)
    f, g = _posterior_mode(lik, n, K, μ, y)
    ll = loglik(lik, f, y)
    H, Hback = Zygote.pullback(hess_loglik, lik, f, y)
    B = lu(I - (K * H))
    L = (0.5 * (g ⋅ (f - μ))) + (0.5 * logdet(B)) - ll
    return L, g, H, Hback, B
end

# Fisher scoring:
# maximize: loglik(f, y) - 0.5 * ((f - μ)' * inv(K) * (f - μ))
# gradient: g = dloglik(f, y) - inv(K) * (f - μ)
# quasi-Hessian: H = - (W + inv(K))    where W is fisher info
# Newton step: 
#  f_new = f + Δf
#  Δf = - inv(H) * g
#   = inv(W + inv(K)) * (dloglik(f, y) - inv(K)*(f-μ))
#   = inv(W + inv(K)) * inv(K) * ((K * dloglik(f, y)) - f + μ)
#   = inv(I + K W) * ((K * dloglik(f, y)) - f + μ)
#
# Note that at the mode (where g=0), we have:
#  f - μ = K * dloglik(f, y)
#
# In _laplace_nlml that lets us avoid inv(K):
#  (f - μ)' inv(K) (f - μ) = dloglik(f, y) ⋅ (f - μ)
function _posterior_mode(lik, n, K, m, y; maxiter=500)
    f = repeat(init_latent(lik, y); inner=n)
    g, ng = _posterior_mode_grads(lik, f, K, m, y)
    α = 1.0
    for i in 1:maxiter
        f_new = f + (α * ng)
        if isapprox(f, f_new)
            # println("converged on iteration $i")
            break
        end            
        g_new, ng_new = _posterior_mode_grads(lik, f_new, K, m, y)
        if (ng ⋅ ng_new) > 0
            f = f_new
            g = g_new
            ng = ng_new
            α = 1.0
        else
            α = 0.5 * α
        end
    end
    return f, g
end

function _posterior_mode_grads(lik, f, K, m, y)
    g = grad_loglik(lik, f, y)
    W = fisher_info(lik, f, y)
    B = lu(I + (K * W))
    ng = B \ ((K * g) - f + m)
    return g, ng
end

function ChainRulesCore.rrule(::typeof(_laplace_nlml), lik, n, K, μ, y)
    L, g, H, Hback, B = _laplace_nlml_and_intermediates(lik, n, K, μ, y)
    function _laplace_nlml_pullback(ΔL)
        Δlik = @not_implemented("gradient of nlml wrt lik params")
        Δn = @not_implemented("gradient of nlml wrt n")
        Δy = @not_implemented("gradient of nlml wrt obs")
        Δμ = ΔL * g
        ΔK = 0.5 * ΔL * ((g * g') + (B' \ H))
        ΔH = 0.5 * ΔL * (B \ K)'
        Δf = Hback(ΔH)[2]
        Bf = (B' \ Δf)
        Δμ += Bf
        ΔK += Bf * g'
        return NoTangent(), Δlik, Δn, -ΔK, -Δμ, Δy
    end
    return L, _laplace_nlml_pullback
end

function predict_latent(gpfit::LaplaceGPRegression, xtest::AbstractMatrix)
    N = size(xtest, 1)
    C = nparam(gpfit.lik)
    Kstars = kernelmatrix.(gpfit.kernels, Ref(RowVecs(gpfit.x)), Ref(RowVecs(xtest)))
    Kstar = cat(Kstars...; dims=(1,2))
    μstar = vcat(fill.(gpfit.params.mean, N)...)
    mean = μstar + (Kstar' * gpfit.g)
    mean = reshape(mean, N, C)
    means = eachrow(mean)
    Kstarstars = kernelmatrix_diag.(gpfit.kernels, Ref(RowVecs(xtest)))
    Kstarstar = Diagonal(vcat(Kstarstars...))
    cov = Kstarstar + (Kstar' * gpfit.HinvB * Kstar)
    cov = reshape(cov, N, C, N, C)
    # Note we force symmetry to deal with slight rounding errors
    covs = [Symmetric(cov[i, :, i, :]) for i=1:N]
    return MvNormal.(means, covs)
end

function predict(gpfit::LaplaceGPRegression, xtest::AbstractMatrix)
    latents = predict_latent(gpfit, xtest)
    return postpred.(Ref(gpfit.lik), latents)
end
