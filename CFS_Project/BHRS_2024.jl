#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Final Project: Unemployment risk in a Life-cycle Bewely economy

    Last Edit:  April 19, 2025
    Authors:    Cutberto Frias Sarraf

=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, LaTeXStrings, Distributions

include("Tauchen_1986_Grid.jl")
include("Tauchen_1986_Drift.jl")

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
    a_max::Float64  = 1500000                       # Maximum value of assets
    na::Int64       = 5000                          # Grid points for assets
    a_grid::Vector{Float64} = exp.(collect(range(log(a_min), length = na, stop = log(a_max))))   

    # Employment/Unemployment
    ne::Int64       = 2
    e_grid::Vector{Float64} = [1, 2]
    # δ::Matrix{Float64} = [0.95 0.05; 0.50 0.50]     # Employment/Unemployment transition matrix
    # P_monthly = [0.985 0.015;
    #          0.2347 0.7653]
    # P_annual = P_monthly^12
    δ::Matrix{Float64} = [0.94184   0.058160;
                          0.910011  0.0899890]
    # δ::Matrix{Float64} = [0.8336 0.1664; 0.9395 0.0605] # Employment/Unemployment transition matrix
    
    # Income process
    ρ::Float64      = 0.97                          # Correlation in the persistent component of income

    nζ::Int64       = 21                            # Grid points for the permanent component
    σ_ζE::Float64   = sqrt(0.08)                    # Standard deviation of the permanent shock in Employment
    σ_ζU::Float64   = sqrt(0.33)                    # Standard deviation of the permanent shock in Unemployment
    B_N::Float64    = -0.18                         # Drift of persistent component while Unemployed
    
    nϵ::Int64       = 11                            # Grid points for the transitory component
    σ_ϵ::Float64    = sqrt(0.04)                    # Standard deviation of the transitory shock


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
    V::Array{Float64,5}
    a_policy::Array{Float64,5}
    a_policy_index::Array{Int64,5}
    c_policy::Array{Float64,5}
end

@with_kw struct OtherPrimitives
    ζ_grid::Vector{Float64}
    T_ζE::Matrix{Float64}
    T_ζU::Matrix{Float64}
    ϵ_grid::Vector{Float64}
    T_ϵ::Vector{Float64}
end

# Function for initializing model primitives and results
function Initialize_Model()
    param = Primitives()
    @unpack_Primitives param

    V                   = zeros(T + 1, na, nζ, nϵ, ne)          # Value function
    a_policy            = zeros(T, na, nζ, nϵ, ne)              # Savings function
    a_policy_index      = zeros(T, na, nζ, nϵ, ne)
    c_policy            = zeros(T, na, nζ, nϵ, ne)              # Consumption function

    ζ_grid, T_ζE        = tauchen(nζ, ρ, σ_ζE)                  # Discretization of Permanent shocks: Employment   [ζE]
    # ζ_grid, T_ζU       = tauchen(nζ, ρ, σ_ζU, grid = ζ_grid)   # Discretization of Permanent shocks: Unemployment [ζU]
    ζ_grid, T_ζU        = tauchen_drift(nζ, ρ, σ_ζU, B = B_N, grid = ζ_grid)   # Discretization of Permanent shocks: Unemployment [ζU] DRIFT
    ϵ_grid, T_ϵ         = tauchen(nϵ, 0.0, σ_ϵ)                 # Discretization of Transitory shocks [ϵ]
    T_ϵ                 = T_ϵ[1,:]
    
    other_param         = OtherPrimitives(ζ_grid, T_ζE, T_ζU, ϵ_grid, T_ϵ)
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
        Y = 0.50 * exp(κ_R)

        # Compute once for a representative ζ and ϵ 
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

                EV  = V[j+1, ap_index, 1, 1, 1]            # Use any index, e.g., 1, 1 and 1
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

        # Copy results across all ζ_index and ϵ_index 
        for ζ_index in 1:nζ                                         # State: Permanent shock ζ 
            for ϵ_index in 1:nϵ                                     # State: Transitory shock ϵ
                for e_index in 1:ne                                 # State: Employment Status e
                    @inbounds for a_index in 1:na                   # Control: Assets a'
                        V[j, a_index, ζ_index, ϵ_index, e_index]                 = temp_V[a_index]
                        c_policy[j, a_index, ζ_index, ϵ_index, e_index]          = temp_c_policy[a_index]
                        a_policy[j, a_index, ζ_index, ϵ_index, e_index]          = temp_a_policy[a_index]
                        a_policy_index[j, a_index, ζ_index, ϵ_index, e_index]    = temp_a_policy_index[a_index]
                    end
                end
            end
        end
    end
    end

    # [2] Working households
    @inbounds begin
    for j in TR-1:-1:1
        println("Age is ", 24+j)
        κ_j = κ[j]

        Tϵ_weighted_VE = zeros(na, nζ)
        Tϵ_weighted_VU = zeros(na, nζ)
        for ap_index in 1:na
            for ζp_index in 1:nζ
                Tϵ_weighted_VE[ap_index, ζp_index] = dot(T_ϵ, V[j+1, ap_index, ζp_index, :, 1])
                Tϵ_weighted_VU[ap_index, ζp_index] = dot(T_ϵ, V[j+1, ap_index, ζp_index, :, 2])
            end
        end
    
    #= --------------------------------- STATE VARIABLES ----------------------------------------- =#
        # State: Employment 
        Threads.@threads for ζ_index in 1:nζ                    # State: Permanent shock ζ 
            ζ = ζ_grid[ζ_index]
                
            for ϵ_index in 1:nϵ                                 # State: Transitory shock ϵ
                ϵ = ϵ_grid[ϵ_index]
                Y = exp(κ_j + ζ + ϵ)                            # Income in levels 
    
                start_index = 1                                 # Use that a'(a) is a weakly increasing function. 
                for a_index in 1:na                             # State: Assets a
                    a = a_grid[a_index]
                    X = Y + (1 + r) * a
    
                    candidate_max = -Inf
    #= --------------------------------- DECISION VARIABLES -------------------------------------- =#
                    for ap_index in start_index:na              # Control: Assets a'
                        ap = a_grid[ap_index]
                        c  = X - ap                             # Consumption
    #= --------------------------------- GRID SEARCH --------------------------------------------- =#
                        if c <= 0                               # Feasibility check
                            continue
                        end
    
                        # Compute expected value from precomputed terms
                        EVEmployment   = δ[1,1] * dot(T_ζE[ζ_index, :], Tϵ_weighted_VE[ap_index, :])
                        EVUnemployment = δ[1,2] * dot(T_ζU[ζ_index, :], Tϵ_weighted_VU[ap_index, :])

                        val = Flow_Utility(c, param) + β * (EVEmployment + EVUnemployment) # Utility Value
    
                        if val > candidate_max                  # Check for max
                            candidate_max                                       = val
                            a_policy[j, a_index, ζ_index, ϵ_index, 1]           = ap
                            c_policy[j, a_index, ζ_index, ϵ_index, 1]           = c
                            a_policy_index[j, a_index, ζ_index, ϵ_index, 1]     = ap_index
                            V[j, a_index, ζ_index, ϵ_index, 1]                  = val
                            start_index                                         = ap_index
                        else
                            break  # Early stopping due to concavity
                        end  
                    end                   
                end 
            end 
        end

        #= --------------------------------- STATE VARIABLES ----------------------------------------- =#
        # State: Unemployment 
        Threads.@threads for ζ_index in 1:nζ                    # State: Permanent shock ζ                 
            for ϵ_index in 1:nϵ                                 # State: Transitory shock ϵ
                Y = 0.50 * exp(κ_j)                             # Income in levels 
    
                start_index = 1                                 # Use that a'(a) is a weakly increasing function. 
                for a_index in 1:na                             # State: Assets a
                    a = a_grid[a_index]
                    X = Y + (1 + r) * a
    
                    candidate_max = -Inf
        #= --------------------------------- DECISION VARIABLES -------------------------------------- =#
                    for ap_index in start_index:na              # Control: Assets a'
                        ap = a_grid[ap_index]
                        c  = X - ap                             # Consumption
        #= --------------------------------- GRID SEARCH --------------------------------------------- =#
                        if c <= 0                               # Feasibility check
                            continue
                        end

                        # Compute expected value from precomputed terms
                        EVEmployment   = δ[2,1] * dot(T_ζE[ζ_index, :], Tϵ_weighted_VE[ap_index, :])
                        EVUnemployment = δ[2,2] * dot(T_ζU[ζ_index, :], Tϵ_weighted_VU[ap_index, :])

                        val = Flow_Utility(c, param) + β * (EVEmployment + EVUnemployment) # Utility Value

                        if val > candidate_max                  # Check for max
                            candidate_max                                       = val
                            a_policy[j, a_index, ζ_index, ϵ_index, 2]           = ap
                            c_policy[j, a_index, ζ_index, ϵ_index, 2]           = c
                            a_policy_index[j, a_index, ζ_index, ϵ_index, 2]     = ap_index
                            V[j, a_index, ζ_index, ϵ_index, 2]                  = val
                            start_index                                         = ap_index
                        else
                            break  # Early stopping due to concavity
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

# using Serialization  # Import the module
# # Saving data
# serialize("CFS_Project/Param_20250419.jls", param)                  # Save the 'prim' variable
# serialize("CFS_Project/Results_20250419.jls", results)              # Save the 'res' variable
# serialize("CFS_Project/OtherParam_20250419.jls", other_param)       # Save the 'prim' variable

# # Loading data
# param      = deserialize("CFS_Project/Param_20250419.jls")         # Load the 'prim' variable
# results    = deserialize("CFS_Project/Results_20250419.jls")       # Load the 'res' variable
# other_param = deserialize("CFS_Project/OtherParam_20250419.jls")    # Load the 'prim' variable
# @unpack_Primitives param                                             
# @unpack_Results results
# @unpack_OtherPrimitives other_param

#= ################################################################################################## 
    Simulations
=# ##################################################################################################

function simulate_model(param, results, other_param, S::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 
    @unpack_Primitives param                                             
    @unpack_Results results
    @unpack_OtherPrimitives other_param

    # [1] Distribution over initial Employment Status
    Initial_dist_E      = Categorical([0.909,0.091])
    Employment_dist     = Categorical(δ[1,:])
    Unemployment_dist   = Categorical(δ[2,:])

    # [2] Distribution over the initial permanent component
    ζ0_grid, T_ζ0       = tauchen(nζ, 0.0, 0.15)                  # Discretization of Initial Permanent shocks  [ζ]    
    Initial_dist_ζ      = Categorical(T_ζ0[1,:])
    Perm_distsE         = [Categorical(T_ζE[i, :]) for i in 1:nζ] # State-contingent distributions over the permanent components
    Perm_distsU         = [Categorical(T_ζU[i, :]) for i in 1:nζ] # State-contingent distributions over the permanent components

    # [3] Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    Transitory_dist     = Categorical(T_ϵ)

    # Outputs
    Assets          = zeros(S,T)                                                # Saving by Age
    Consumption     = zeros(S,T) 
    Persistent      = zeros(S,T)
    Transitory      = zeros(S,T)
    Income          = zeros(S,T) 
    Employment      = zeros(S,T) 


    for s = 1:S
        employment_index    = rand(Initial_dist_E)
        persistent_index    = rand(Initial_dist_ζ)
        transitory_index    = rand(Transitory_dist)
        a_index             = 1                                                 # Start with 0 assets

        # Employment Status
        Employment[s,1]     = employment_index

        # Asset policy 
        Assets[s,1]         = a_policy[1, a_index, persistent_index, transitory_index, employment_index]

        # Persistent and Transitory components 
        Persistent[s,1]     = ζ_grid[persistent_index]
        Transitory[s,1]     = ϵ_grid[transitory_index]

        # Compute income
        if employment_index == 1
            Income[s,1]     = exp(κ[1] + Persistent[s,1] + Transitory[s,1]) 
        else 
            Income[s,1]     = 0.50 * exp(κ[1])
        end

        # Consumption policy 
        Consumption[s,1]    = a_grid[a_index]*(1+r) + Income[s,1] - Assets[s,1]

        for t = 2:T 

            # Given Employment Status in t-1 we draw the Employment Status in t
            if employment_index == 1
                employment_index = rand(Employment_dist)
            else 
                employment_index = rand(Unemployment_dist)
            end
            Employment[s,t]      = employment_index

            # Given the Employment Status in t, we draw the new persistent and transitory shocks
            if employment_index == 1
                persistent_index = rand(Perm_distsE[persistent_index])          # Draw the new permanent component based upon the old one. 
            else 
                persistent_index = rand(Perm_distsU[persistent_index])           
            end
            transitory_index     = rand(Transitory_dist)                        # Draw the transitory component 
            a_index              = findfirst(x -> x == Assets[s, t-1], a_grid)  # Find the index of the previous choice of ap 

            # Outputs 
            Persistent[s,t]     = ζ_grid[persistent_index]
            Transitory[s,t]     = ϵ_grid[transitory_index]
            Assets[s,t]         = a_policy[t, a_index, persistent_index, transitory_index, employment_index]

            if t < TR
                if employment_index == 1
                    Income[s,t] = exp(κ[t] + Persistent[s,t] + Transitory[s,t])
                else 
                    Income[s,t] = 0.50 * exp(κ[t])
                end
            else
                    Income[s,t] = 0.50 * exp(κ[TR-1])
            end

            Consumption[s,t]    = a_grid[a_index] * (1+r) + Income[s,t] - Assets[s,t]
        end 
    end 

    return Assets, Consumption, Persistent, Transitory, Income, Employment
end


function True_insurance(Transitory::Matrix{Float64}, Persistent::Matrix{Float64}, Consumption::Matrix{Float64}, Employment::Matrix{Float64}, ρ::Float64, TR::Int64)
    S = size(Consumption, 1)

    log_consumption     = log.(Consumption)[:,1:TR-1]
    Employment_Status   = Employment[:,2:TR-1]
    consumption_growth  = diff(log_consumption, dims = 2)

    # Initialize values
    covs_cϵ     = 0
    var_ϵ       = 0
    covs_cζE    = 0
    var_ζE      = 0
    covs_cζU    = 0
    var_ζU      = 0

    # Vectors 
    ζ = Persistent[:,2:TR-1] .- ρ .* Persistent[:,1:TR-2]
    ϵ = Transitory[:,2:TR-1]
    Vector_Employment   = vec(Employment_Status)
    Vector_ζ            = vec(ζ) 
    Vector_Δc           = vec(consumption_growth)
    Vector_ϵ            = vec(ϵ) 

    # Conditional Vectors
    Vector_ζ_E  = Vector_ζ[Vector_Employment .== 1.0]
    Vector_ϵ_E  = Vector_ϵ[Vector_Employment .== 1.0]
    Vector_ζ_U  = Vector_ζ[Vector_Employment .== 2.0]
    Vector_Δc_E = Vector_Δc[Vector_Employment .== 1.0]
    Vector_Δc_U = Vector_Δc[Vector_Employment .== 2.0]
    
    # Insurance Coefficient: Transitory
    covs_cϵ = cov(Vector_Δc_E, Vector_ϵ_E)
    var_ϵ   = var(Vector_ϵ_E)
    α_ϵ     = 1 - covs_cϵ / var_ϵ

    # Insurance Coefficient: Employment Persistent 
    covs_cζE = cov(Vector_Δc_E, Vector_ζ_E)
    var_ζE   = var(Vector_ζ_E)
    α_ζE     = 1- covs_cζE / var_ζE

    # Insurance Coefficient: Unemployment Persistent 
    covs_cζU = cov(Vector_Δc_U, Vector_ζ_U)
    var_ζU   = var(Vector_ζ_U)
    α_ζU     = 1- covs_cζU / var_ζU

    return α_ϵ, α_ζE, α_ζU
end

function True_insurance_by_age(Transitory::Matrix{Float64}, Persistent::Matrix{Float64}, Consumption::Matrix{Float64}, Employment::Matrix{Float64}, ρ::Float64, TR::Int64)
    S = size(Consumption, 1)

    log_consumption     = log.(Consumption)[:,1:TR-1]
    Employment_Status   = Employment[:,2:TR-1]
    consumption_growth  = diff(log_consumption, dims = 2)

    # Initialize vectors to hold age-specific coefficients
    α_ϵ         = zeros(TR-2)
    α_ζE        = zeros(TR-2)
    α_ζU        = zeros(TR-2)
    covs_cϵ     = Float64[]
    var_ϵ       = Float64[]
    covs_cζE    = Float64[]
    var_ζE      = Float64[]
    covs_cζU    = Float64[]
    var_ζU      = Float64[]

    Vector_ζ_E  = Float64[]
    Vector_ϵ_E  = Float64[]
    Vector_ζ_U  = Float64[]
    Vector_Δc_E = Float64[]
    Vector_Δc_U = Float64[]

    # Vectors 
    ζ  = Persistent[:,2:TR-1] .- ρ .* Persistent[:,1:TR-2]
    ϵ  = Transitory[:,2:TR-1]
    Δc = consumption_growth

    # Transitory and Persistent insurance by age
    for t = 1:TR-2

        # Conditional Vectors
        Vector_ζ_E  = ζ[:,t][Employment_Status[:,t] .== 1.0]
        Vector_ϵ_E  = ϵ[:,t][Employment_Status[:,t] .== 1.0]
        Vector_ζ_U  = ζ[:,t][Employment_Status[:,t] .== 2.0]
        Vector_Δc_E = Δc[:,t][Employment_Status[:,t] .== 1.0]
        Vector_Δc_U = Δc[:,t][Employment_Status[:,t] .== 2.0]
        
        covs_cϵ     = cov(Vector_Δc_E, Vector_ϵ_E)     # Size S x T-1 and S x T-1
        var_ϵ       = var(Vector_ϵ_E)
        α_ϵ[t]      = 1 - covs_cϵ / var_ϵ

        covs_cζE    = cov(Vector_Δc_E, Vector_ζ_E)
        var_ζE      = var(Vector_ζ_E)
        α_ζE[t]     = 1 - covs_cζE / var_ζE

        covs_cζU    = cov(Vector_Δc_U, Vector_ζ_U)
        var_ζU      = var(Vector_ζ_U)
        α_ζU[t]     = 1 - covs_cζU / var_ζU

    end

    return α_ϵ, α_ζE, α_ζU
end

#= ################################################################################################## 
    Plots: Policy Functions
=# ##################################################################################################

# Employed Value Policy Function: Assets
age       = [25, 35, 45, 55, 65, 75, 85, 95]
indices   = [ 1, 11, 21, 31, 41, 51, 61, 70]
plot(a_grid/1000, V[indices[1], :, 11, 6, 1], label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, V[idx, :, 11, 6, 1], label = "Age = $t")
end
title!("Employed Value function")
xlabel!("Assets a (\$1000)")
ylabel!("Value function")
plot!(legend=:bottomright)

# Savings Policy Function: Assets
age       = [25, 35, 45, 55, 65, 75, 85, 95]
indices   = [ 1, 11, 21, 31, 41, 51, 61, 70]
plot(a_grid/1000, a_policy[indices[1], :, 11, 6, 1]/1000, label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, a_policy[idx, :, 11, 6, 1]/1000, label = "Age = $t")
end
title!("Employed Asset Policy Function")
xlabel!("Assets a (\$1000)")
ylabel!("Assets a' (\$1000)")
plot!(legend=:bottomright)

# Consumption Policy Function: Assets
age       = [25, 35, 45, 55, 65, 75, 85]
indices   = [ 1, 11, 21, 31, 41, 51, 61]
plot(a_grid/1000, c_policy[indices[1], :, 11, 6, 1]/1000, label = "Age = $(age[1])")
for (t, idx) in zip(age[2:end], indices[2:end])
    plot!(a_grid/1000, c_policy[idx, :, 11, 6, 1]/1000, label = "Age = $t")
end
title!("Employed Consumption Policy Function")
xlabel!("Assets a (\$1000)")
ylabel!("Consumption c (\$1000)")
plot!(legend=:bottomright)

#= ################################################################################################## 
    Plots: Simulations
=# ##################################################################################################

age_grid_short = collect(Int, range(25, 59, length=35)) 
age_grid_full  = collect(Int, range(25, 94, length=70)) 

S        = 50000
Assets, Consumption, Persistent, Transitory, Income, Employment = simulate_model(param, results, other_param, S)
α_ϵ, α_ζE, α_ζU                                                 = True_insurance(Transitory, Persistent, Consumption, Employment, ρ, TR)
Vector_α_ϵ, Vector_α_ζE, Vector_α_ζU                            = True_insurance_by_age(Transitory, Persistent, Consumption, Employment, ρ, TR)

#####################################################################################################

# WEALTH BY AGE: WORKING PHASE
plot(age_grid_short, vec(mean(Assets,dims = 1))[1:35] ./1000, label = "Mean Wealth")
plot!(age_grid_short, vec(median(Assets,dims = 1))[1:35] ./1000,label = "Median Wealth")
title!("Wealth Accumulation over the Lifecycle")
xlabel!("Age")
ylabel!(L"Wealth ($1000)")
plot!(legend=:topleft)
# savefig("CFS_Project/Project_Image_01.png") 

# WEALTH BY AGE: WORKING AND RETIREMENT PHASE
plot(age_grid_full, vec(mean(Assets,dims = 1))/1000, label = "Mean Wealth")
plot!(age_grid_full, vec(median(Assets,dims = 1))/1000, label = "Median Wealth")
title!("Wealth Accumulation over the Lifecycle")
xlabel!("Age")
ylabel!(L"Wealth ($1000)")
plot!(legend=:topleft)

# INCOME BY AGE
plot(age_grid_full, vec(mean(Income,dims = 1))/1000, label = "Mean Income")
plot!(age_grid_full, vec(median(Income,dims = 1))/1000, label = "Median Income")
title!("Income over the Lifecycle")
xlabel!("Age")
ylabel!(L"Income ($1000)")
plot!(legend=:topleft)

# CONSUMPTION BY AGE
plot(age_grid_full, vec(mean(Consumption,dims = 1))/1000, label = "Mean Consumption")
plot!(age_grid_full, vec(median(Consumption,dims = 1))/1000, label = "Median Consumption")
title!("Consumption over the Lifecycle")
xlabel!("Age")
ylabel!(L"Consumption ($1000)")
plot!(legend=:topleft)

# Plot consumption std dev by age 
plot(age_grid_full, vec(std(Consumption, dims = 1))/1000, label = "Std Dev Consumption")
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
    normalize  = :probability,
    size       = (400, 500))
savefig("CFS_Project/Figures/Project_Image_BHRS_02.png") 

# Compute mean over S
mean_Labor_income = vec(mean(Income, dims=1))/1000
mean_Consumption  = vec(mean(Consumption,  dims=1))/1000
# mean_Wealth       = vec(mean(Assets,  dims=1))/1000
mean_Wealth       = vcat(0.0, vec(mean(Assets,  dims=1))[1:end-1]/1000)

# Figure 3A: Income, Consumption and Wealth 
plot(age_grid_full, mean_Consumption, title = "Dynamics over the Life Cycles", ylabel = "(\$1000)", 
    label = "Consumption" , xlabel = "Age")
plot!(age_grid_full, mean_Labor_income, title = "Dynamics over the Life Cycles", ylabel = "(\$1000)", 
label = "Net Income" , xlabel = "Age")
plot!(age_grid_full, mean_Wealth, title = "Dynamics over the Life Cycles", ylabel = "(\$1000)", 
label = "Wealth" , xlabel = "Age")
plot!(
    legend = :topright,
    xlims = (25, 95),
    ylims = (0, 750),
    xticks = 25:5:95,
    yticks = 0:50:750,
    xtickfont  = font(9),
    ytickfont  = font(9),
    guidefont  = font(11),
    legendfont = font(7),
    size       = (400, 500))     
savefig("CFS_Project/Figures_PPT/BHRS_Dynamics_Image_01.png") 

#####################################################################################################

# Age profiles of insurance coefficients: Transitory shocks
plot(age_grid_short[2:TR-1], Vector_α_ϵ,
     label      = "Model TRUE",
     lw         = 2,                  
     color      = :black)
title!(L"\mathsf{Insurance\ Coefficients:\ Transitory\ Shocks}")
xlabel!("Age")
ylabel!(L"\mathsf{Insurance\ Coefficient}\ \phi^{\varepsilon}")
plot!(legend = :bottomright,             
      xtickfont  = font(9),
      ytickfont  = font(9),
      guidefont  = font(11),
      legendfont = font(7),
      titlefont  = font(12), 
      size       = (400, 500))     
savefig("CFS_Project/Figures_PPT/BHRS_Transitory_Image_03.png") 

# Age profiles of insurance coefficients: Permanent shocks
plot(age_grid_short[2:TR-1], Vector_α_ζE,
     label      = "Model TRUE: Employed",
     lw         = 2,                  
     color      = :black)
# plot!(age_grid_short[2:TR-1], Vector_α_ζU,
#      label      = "Model TRUE: Unemployed",
#      lw         = 2,                  
#      color      = :red)
title!(L"\mathsf{Insurance\ Coefficients:\ Persistent\ Shocks}")
xlabel!("Age")
ylabel!(L"\mathsf{Insurance\ Coefficient}\ \phi^{\eta}")
plot!(legend = :topleft,             
      xtickfont  = font(9),
      ytickfont  = font(9),
      guidefont  = font(11),
      legendfont = font(7),
      titlefont  = font(12), 
      size       = (400, 500)) 
savefig("CFS_Project/Figures_PPT/BHRS_Persistent_Image_04.png") 
