function _init_params(x)
    return (
        mean = 0.0,
        noise_var = positive(1.0),
        k = (
            var = positive(1.0),
            precision = inv.(maximum(x; dims=1) .- minimum(x; dims=1))[:]
        )
    )
end

function _kernel(base_kernel::Kernel, params)
    return params.var * (base_kernel ∘ ARDTransform(params.precision))
end

function _zygote_fg!(f)
    return Optim.only_fg!() do F, G, θ
        if G !== nothing
            val, grad = Zygote.withgradient(f, θ)
            copyto!(G, only(grad))
            if F !== nothing
                return val
            end
        end
        if F !== nothing
            return f(θ)
        end
        return nothing
    end
end