
function True_insurance(Transitory::Matrix{Float64}, Persistent::Matrix{Float64}, Consumption::Matrix{Float64}, ρ::Float64, TR::Float64)
    S = size(Consumption, 1)

    log_consumption     = log.(Consumption)[:,1:TR-1]
    consumption_growth  = diff(log_consumption, dims = 2)

    # Initialize values
    covs_cϵ = 0
    var_ϵ   = 0
    covs_cζ = 0
    var_ζ   = 0

    # Vectors 
    ζ = Persistent[:,2:TR-1] .- ρ .* Persistent[:,1:TR-2]
    ϵ = Transitory[:,2:TR-1]

    Vector_ζ  = vec(ζ) 
    Vector_Δc = vec(consumption_growth)
    Vector_ϵ  = vec(ϵ) 
    
    # Insurance Coefficient: Transitory
    covs_cϵ = cov(Vector_Δc, Vector_ϵ)
    var_ϵ   = var(Vector_ϵ)
    α_ϵ     = 1 - covs_cϵ / var_ϵ

    # Insurance Coefficient: Persistent
    covs_cζ = cov(Vector_Δc, Vector_ζ)
    var_ζ   = var(Vector_ζ)
    α_ζ     = 1- covs_cζ / var_ζ

    return α_ϵ, α_ζ
end




function True_insurance(Transitory::Matrix{Float64}, Persistent::Matrix{Float64}, Consumption::Matrix{Float64}, ρ::Float64)
    S = size(Consumption, 1)
    T = size(Consumption, 2)

    log_consumption     = log.(Consumption)
    consumption_growth  = diff(log_consumption, dims = 2)

    a = zeros(S)
    b = zeros(S)
    for s = 1:S
        a[s] = cov(consumption_growth[s,1:T-1], Transitory[s,2:T])
        b[s] = var(Transitory[s,2:T])
    end 

    α_ϵ = 1 - sum(a) / sum(b)

    ζ =  Persistent[:,2:T] .- ρ .* Persistent[:,1:T-1]
    c = zeros(S)
    d = zeros(S)
    for s = 1:S
        c[s] = cov(consumption_growth[s,1:T-1], ζ[s,:])
        d[s] = var(ζ[s,:])
    end 

    α_ζ = 1- sum(c) / sum(d)
    return α_ϵ,α_ζ
end