#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Final Project: Unemployment risk in a Life-cycle Bewely economy

    Last Edit:  April 17, 2025
    Authors:    Cutberto Frias Sarraf

=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, LaTeXStrings, Distributions

include("Tauchen_Hussey_1991.jl")

#= ################################################################################################## 
    Parameters
=# ##################################################################################################

@with_kw struct Primitives

    T::Int64        = 70                            # Life-cycle 35 Working years
    TR::Int64       = 36                            # Life-cycle 35 Retirement years
    r::Float64      = 0.03                          # Interest rate  
    β::Float64      = 0.975                         # Discount rate  
    γ::Float64      = 2.0                           # Coefficient of Relative Risk Aversion 

    # Assets Grid
    a_min::Float64  = 0.1                           # Minimum value of assets
    a_max::Float64  = 2000000                       # Maximum value of assets
    na::Int64       = 1000                           # Grid points for assets
    a_grid::Vector{Float64} = exp.(collect(range(log(a_min), length = na, stop = log(a_max))))   

    # Income process
    ρ::Float64      = 0.97                          # Correlation in the persistent component of income 
    nζ::Int64       = 11                            # Grid points for the permanent component
    σ_ζ::Float64    = sqrt(0.01)                    # Standard deviation of the permanent shock

    nϵ::Int64       = 6                            # Grid points for the transitory component
    σ_ϵ::Float64    = sqrt(0.05)                    # Standard deviation of the transitory shock

    κ::Vector{Float64} = [                          # Deterministic age profile for log-income
    10.00571682417030, 10.06468173213630, 10.14963371320800, 10.18916005760660, 10.25289993933830,
    10.27787916956560, 10.32260755975800, 10.36733797632800, 10.39391841908670, 10.42305441774350,
    10.45397023113620, 10.48282124181770, 10.50757066459240, 10.53338513735820, 10.53669397036220,
    10.56330457698600, 10.58945748446780, 10.60438525029320, 10.62570875544670, 10.62540348055990,
    10.63732032711820, 10.64422326499790, 10.65124153265610, 10.64869889614020, 10.60836674391850,
    10.61725912807620, 10.60201099108720, 10.58990416581600, 10.55571432462690, 10.54753392025080,
    10.53038700787840, 10.51112990486990, 10.50177243660240, 10.49346004128460, 10.48778926452950
    ]

end 

# Initialize value function and policy functions
@with_kw mutable struct Results
    V::Array{Float64,4}
    a_policy::Array{Float64,4}
    a_policy_index::Array{Int64,4}
    c_policy::Array{Float64,4}
end

@with_kw struct OtherPrimitives
    ζ_grid::Vector{Float64}
    T_ζ::Matrix{Float64}
    ϵ_grid::Vector{Float64}
    T_ϵ::Vector{Float64}
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    V                   = zeros(T + 1, na, nζ, nϵ)          # Value function
    a_policy            = zeros(T, na, nζ, nϵ)              # Savings function
    a_policy_index      = zeros(T, na, nζ, nϵ)
    c_policy            = zeros(T, na, nζ, nϵ)              # Consumption function

    ζ_grid, T_ζ         = tauchen_hussey(nζ, ρ  , σ_ζ)      # Discretization of Permanent shocks  [ζ]
    ϵ_grid, T_ϵ         = tauchen_hussey(nϵ, 0.0, σ_ϵ)      # Discretization of Transitory shocks [ϵ]
    T_ϵ                 = T_ϵ[1,:]
    
    other_param         = OtherPrimitives(ζ_grid, T_ζ, ϵ_grid, T_ϵ)
    results             = Results(V, a_policy, a_policy_index, c_policy)

    return param, results, other_param
end

#= ################################################################################################## 
  
    Functions

=# ##################################################################################################

function Flow_Utility(c::Float64, param::Primitives)
    @unpack_Primitives param                
    return (c^(1 - γ)) / (1 - γ)
end 

function Solve_Problem(param::Primitives, results::Results, other_param::OtherPrimitives)
    # Solves the decision problem: param is the structure of parameters and results stores solutions 
    @unpack_Primitives param                
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    println("Begin solving the model backwards")
    κ_R = κ[TR-1]

    # [1] Retired households
    @inbounds begin
    for j in T:-1:TR 
        println("Age is ", 24+j)

    #= --------------------------------- STATE VARIABLES ----------------------------------------- =#
        for ζ_index in 1:nζ                                 # State: Permanent shock ζ 
            ζ = ζ_grid[ζ_index]
            Y = 0.30 * exp(κ_R + ζ)

            # Compute once for a representative ϵ (e.g. ϵ_index = 1)
            temp_V              = zeros(na)
            temp_c_policy       = zeros(na)
            temp_a_policy       = zeros(na)
            temp_a_policy_index = zeros(Int, na)

            start_index = 1                                 # Use that a'(a) is a weakly increasing function. 

            @inbounds for a_index in 1:na                   # State: Assets a
                a = a_grid[a_index]
                X = Y + (1 + r) * a
    
                candidate_max = -Inf
    #= --------------------------------- DECISION VARIABLES -------------------------------------- =#
                for ap_index in start_index:na
                    ap = a_grid[ap_index]
                    c = X - ap
    #= --------------------------------- GRID SEARCH --------------------------------------------- =#
                    if c <= 0
                        continue
                    end
    
                    EV  = V[j+1, ap_index, ζ_index, 1]      # Use any ϵ_index, e.g., 1
                    val = Flow_Utility(c, param) + β * EV
    
                    if val > candidate_max
                        candidate_max                   = val
                        temp_V[a_index]                 = val
                        temp_c_policy[a_index]          = c
                        temp_a_policy[a_index]          = ap
                        temp_a_policy_index[a_index]    = ap_index
                        start_index                     = ap_index
                    end
                end
            end

            # Copy result across all ϵ_index
            for ϵ_index in 1:nϵ
                @inbounds for a_index in 1:na
                    V[j, a_index, ζ_index, ϵ_index]                 = temp_V[a_index]
                    c_policy[j, a_index, ζ_index, ϵ_index]          = temp_c_policy[a_index]
                    a_policy[j, a_index, ζ_index, ϵ_index]          = temp_a_policy[a_index]
                    a_policy_index[j, a_index, ζ_index, ϵ_index]    = temp_a_policy_index[a_index]
                end
            end
        end
    end
    end

    # [2] Working households
    @inbounds begin
    for j in TR-1:-1:1  # Backward induction
        println("Age is ", 24+j)
        κ_j = κ[j]

    #= --------------------------------- STATE VARIABLES ----------------------------------------- =#
        for ζ_index in 1:nζ                               # State: Permanent shock ζ 
                ζ = ζ_grid[ζ_index]
                
                for ϵ_index in 1:nϵ                           # State: Transitory shock ϵ
                ϵ = ϵ_grid[ϵ_index]
                Y = exp(κ_j + ζ + ϵ)                      # Income in levels 

                start_index = 1                           # Use that a'(a) is a weakly increasing function. 
                @inbounds for a_index in 1:na             # State: Assets a
                    a = a_grid[a_index]
                    X = Y + (1 + r) * a

                    candidate_max = -Inf
    #= --------------------------------- DECISION VARIABLES -------------------------------------- =#
                    for ap_index in start_index:na        # Control: Assets a': 1:na start_index:na
                    ap = a_grid[ap_index]
                    c  = X - ap                           # Consumption
    #= --------------------------------- GRID SEARCH --------------------------------------------- =#
                        if c <= 0                         # Feasibility check
                            continue
                        end

                        # Compute expected value
                        EV = 0.0
                        @inbounds for ζp_index in 1:nζ
                            for ϵp_index in 1:nϵ
                                EV += T_ζ[ζ_index, ζp_index] * T_ϵ[ϵp_index] * V[j+1, ap_index, ζp_index, ϵp_index]
                            end
                        end

                        val = Flow_Utility(c, param) + β * EV  # Utility Value

                        if val > candidate_max                 # Check for max
                            candidate_max                                   = val
                            a_policy[j, a_index, ζ_index, ϵ_index]          = ap
                            c_policy[j, a_index, ζ_index, ϵ_index]          = c
                            a_policy_index[j, a_index, ζ_index, ϵ_index]    = ap_index
                            start_index                                     = ap_index
                            V[j, a_index, ζ_index, ϵ_index]                 = val
                        end                     
                    end 
                end 
            end 
        end
    end
    end 
end

#= ################################################################################################## 
    Solving the Model
=# ##################################################################################################
param, results, other_param = Initialize_Model()
Solve_Problem(param, results, other_param)
@unpack_Primitives param                                             
@unpack_Results results
@unpack_OtherPrimitives other_param

#= ###################################################################################################
    Save Results
=# ###################################################################################################

using Serialization  # Import the module
# Saving data
serialize("CFS_Project/Param_20250419.jls", param)                  # Save the 'prim' variable
serialize("CFS_Project/Results_20250419.jls", results)              # Save the 'res' variable
serialize("CFS_Project/OtherParam_20250419.jls", other_param)       # Save the 'prim' variable

# Loading data
param      = deserialize("CFS_Project/Param_20250419.jls")         # Load the 'prim' variable
results    = deserialize("CFS_Project/Results_20250419.jls")       # Load the 'res' variable
other_param = deserialize("CFS_Project/OtherParam_20250419.jls")    # Load the 'prim' variable
@unpack_Primitives param                                             
@unpack_Results results
@unpack_OtherPrimitives other_param

#= ################################################################################################## 
    Simulations
=# ##################################################################################################

function simulate_model(param, results, other_param, S::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 
    @unpack_Primitives param                                             
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    # Distribution over the initial permanent component
    ζ0_grid, T_ζ0   = tauchen_hussey(nζ, 0.0, 0.15)        # Discretization of Initial Permanent shocks  [ζ]    
    Initial_dist    = Categorical(T_ζ0[1,:])

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    Transitory_dist = Categorical(T_ϵ)

    # State-contingent distributions over the permanent components
    Perm_dists      = [Categorical(T_ζ[i, :]) for i in 1:nζ]

    # Outputs
    Assets          = zeros(S,T)                                                # Saving by Age
    Consumption     = zeros(S,T) 
    Persistent      = zeros(S,T)
    Transitory      = zeros(S,T)
    Income          = zeros(S,T) 

    for s = 1:S
        transitory_index    = rand(Transitory_dist)
        persistent_index    = rand(Initial_dist)

        a_index             = 1                                                 # Start with 0 assets

        # Asset policy 
        Assets[s,1]         = a_policy[1, a_index, persistent_index, transitory_index]

        # Persistent and Transitory components 
        Persistent[s,1]     = ζ_grid[persistent_index]
        Transitory[s,1]     = ϵ_grid[transitory_index]

        # Compute income
        Income[s,1]         = exp( κ[1] + Persistent[s,1] + Transitory[s,1]) 

        # Consumption policy 
        Consumption[s,1]    = a_grid[a_index]*(1+r) + Income[s,1] - Assets[s,1]

        for t = 2:T 
            persistent_index    = rand(Perm_dists[persistent_index])           # Draw the new permanent component based upon the old one. 
            transitory_index    = rand(Transitory_dist)                        # Draw the transitory component 
            a_index             = findfirst(x -> x == Assets[s, t-1], a_grid)  # Find the index of the previous choice of ap 

            # Outputs 
            Assets[s,t]         = a_policy[t, a_index, persistent_index, transitory_index]
            Persistent[s,t]     = ζ_grid[persistent_index]
            Transitory[s,t]     = ϵ_grid[transitory_index]
            if t < TR
                Income[s,t]     = exp(κ[t] + Persistent[s,t] + Transitory[s,t])
            else
                Income[s,t]     = 0.5 * exp(κ[TR-1] + Persistent[s,TR-1])
            end
            Consumption[s,t]    = a_grid[a_index] * (1+r) + Income[s,t] - Assets[s,t]
        end 
    end 

    return Assets, Consumption, Persistent, Transitory, Income
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

function True_insurance_by_age(Transitory::Matrix{Float64}, Persistent::Matrix{Float64}, Consumption::Matrix{Float64}, ρ::Float64)
    S, T                = size(Consumption)
    log_consumption     = log.(Consumption)
    consumption_growth  = diff(log_consumption, dims = 2)               # Size S x (T-1)

    # Initialize vectors to hold age-specific coefficients
    α_ϵ     = zeros(T - 1)
    α_ζ     = zeros(T - 1)
    covs_cϵ = zeros(T - 1)
    var_ϵ   = zeros(T - 1)
    covs_cζ = zeros(T - 1)
    var_ζ   = zeros(T - 1)

    # Transitory innovation
    ϵ = Transitory[:,2:T]

    # Transitory insurance by age
    for t = 2:T
        covs_cϵ  = cov(consumption_growth[:,t-1], ϵ[:,t-1])     # Size S x T-1 and S x T-1
        var_ϵ    = var(ϵ[:,t-1])
        α_ϵ[t-1] = 1 - covs_cϵ / var_ϵ
    end

    # Persistent innovation: ζₛₜ = Zₛₜ − ρ * Zₛₜ₋₁
    ζ = Persistent[:,2:T] .- ρ .* Persistent[:,1:T-1]

    # Persistent insurance by age
    for t = 2:T
        covs_cζ  = cov(consumption_growth[:,t-1], ζ[:,t-1])
        var_ζ    = var(ζ[:,t-1])
        α_ζ[t-1] = 1 - covs_cζ / var_ζ
    end

    return α_ϵ, α_ζ, covs_cϵ, var_ϵ, covs_cζ, var_ζ
end

#= ################################################################################################## 
    Plots: Policy Functions
=# ##################################################################################################

# Value Policy Function: Assets
age       = [25, 35, 45, 55, 65, 75, 85, 95]
indices   = [ 1, 11, 21, 31, 41, 51, 61, 70]
plot(a_grid/1000, V[indices[1], :, 11, 6], label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, V[idx, :, 11, 6], label = "Age = $t")
end
title!("")
xlabel!("Assets a (\$1000)")
ylabel!("Value function")
plot!(legend=:bottomright)

# Savings Policy Function: Assets
age       = [25, 35, 45, 55, 65, 75, 85, 95]
indices   = [ 1, 11, 21, 31, 41, 51, 61, 70]
plot(a_grid/1000, a_policy[indices[1], :, 11, 6]/1000, label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, a_policy[idx, :, 11, 6]/1000, label = "Age = $t")
end
title!("")
xlabel!("Assets a (\$1000)")
ylabel!("Assets a' (\$1000)")
plot!(legend=:bottomright)

# Consumption Policy Function: Assets
age       = [25, 35, 45, 55, 65, 75, 85, 95]
indices   = [ 1, 11, 21, 31, 41, 51, 61, 70]
plot(a_grid/1000, c_policy[indices[1], :, 11, 6]/1000, label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, c_policy[idx, :, 11, 6]/1000, label = "Age = $t")
end
title!("")
xlabel!("Assets a (\$1000)")
ylabel!("Consumption c (\$1000)")
plot!(legend=:bottomright)

#= ################################################################################################## 
    Plots: Simulations
=# ##################################################################################################

age_grid_short = collect(Int, range(25, 59, length=35)) 
age_grid_full  = collect(Int, range(25, 94, length=70)) 

S        = 50000
Assets, Consumption, Persistent, Transitory, Income     = simulate_model(param, results, other_param, S)
α_ϵ, α_ζ                                                = True_insurance(Transitory, Persistent, Consumption, ρ)
Vector_α_ϵ, Vector_α_ζ, covs_cϵ, var_ϵ, covs_cζ, var_ζ  = True_insurance_by_age(Transitory, Persistent, Consumption, ρ)

#####################################################################################################

# Plot wealth statistics by age 
plot(age_grid_short, vec(mean(Assets,dims = 1))[1:35] ./1000, label = "Mean Wealth")
plot!(age_grid_short, vec(median(Assets,dims = 1))[1:35] ./1000,label = "Median Wealth")
title!("Wealth Accumulation over the Lifecycle")
xlabel!("Age")
ylabel!(L"Wealth ($1000)")
plot!(legend=:topleft)
# savefig("CFS_Project/Project_Image_01.png") 

plot(age_grid_full, vec(mean(Assets,dims = 1))/1000, label = "Mean Wealth")
plot!(age_grid_full, vec(median(Assets,dims = 1))/1000, label = "Median Wealth")
title!("Wealth Accumulation over the Lifecycle")
xlabel!("Age")
ylabel!(L"Wealth ($1000)")
plot!(legend=:topleft)

# Plot consumption std dev by age 
plot(age_grid_full, vec(std(Consumption, dims = 1)), label = "Std Dev Consumption")
title!("Consumption Inequality by Age")
xlabel!("Age")
ylabel!(L"Standard Deviation($1000)")
plot!(legend=:topleft)
# savefig("CFS_Project/Project_Image_02.png") 

# Histogram of Assets
histogram(vec(Assets)/1000,
    label      = "",
    title      = "Distribution of Wealth",
    xlabel     = "Wealth (\$1000)",
    bins       = 100, 
    legend     = :topleft,
    fillalpha  = 0.6,       # Controls fill transparency
    fillcolor  = :blue,     # Optional: choose your fill color
    linecolor  = :black,    # Optional: border of each bar
    normalize  = :probability)
# savefig("CFS_Project/Project_Image_03.png") 


# Age profiles of insurance coefficients: Transitory shocks
plot(age_grid[2:T-4], Vector_α_ϵ[1:T-5],
     label      = "True Model",
     lw         = 2,                  
     color      = :black)
title!(L"\mathsf{Insurance\ Coefficients:\ Transitory\ Shocks}")
xlabel!("Age")
ylabel!(L"\mathsf{Insurance\ Coefficient}\ \phi^{\varepsilon}")
plot!(legend = :topleft,             
      xtickfont  = font(9),
      ytickfont  = font(9),
      guidefont  = font(11),
      legendfont = font(9),
      size       = (500, 500))     

# Age profiles of insurance coefficients: Permanent shocks
plot(age_grid[2:T-4], Vector_α_ζ[1:T-5],
     label      = "True Model",
     lw         = 2,                  
     color      = :black)
title!(L"\mathsf{Insurance\ Coefficients:\ Permanent\ Shocks}")
xlabel!("Age")
ylabel!(L"\mathsf{Insurance\ Coefficient}\ \phi^{\eta}")
plot!(legend = :topleft,             
      xtickfont  = font(9),
      ytickfont  = font(9),
      guidefont  = font(11),
      legendfont = font(9),
      size       = (500, 500)) 