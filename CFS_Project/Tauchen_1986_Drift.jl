
using Distributions

function tauchen_drift(N::Int, ρ::Float64, σ::Float64; μ::Float64=0.0, B::Float64=0.0, m::Float64=3.0, grid::Union{Nothing, Vector{Float64}}=nothing)
    """
    Discretizes an AR(1) process: z' = ρ * z + B + ν, ν ~ N(0, σ^2)

    Arguments:
        N    : Number of grid points
        ρ    : AR(1) persistence parameter
        σ    : Standard deviation of ν
        μ    : Unconditional mean of the process (default = 0)
        B    : Drift term added to the AR(1) process (default = 0)
        m    : Width parameter for grid range (default = 3 standard deviations)
        grid : Optional user-supplied grid vector

    Returns:
        z_grid : Vector of grid points
        P      : Transition matrix (N x N)
    """

    # Use supplied grid or construct default grid
    if isnothing(grid)
        σ_z = σ / sqrt(1 - ρ^2)
        z_min = μ - m * σ_z
        z_max = μ + m * σ_z
        z_grid = range(z_min, z_max, length=N) |> collect
    else
        z_grid = grid
    end

    Δ = z_grid[2] - z_grid[1]
    P = zeros(N, N)

    for j in 1:N
        for k in 1:N
            μ_cond = ρ * z_grid[j] + B  # conditional mean with drift B

            if k == 1
                P[j, k] = cdf(Normal(μ_cond, σ), z_grid[k] + Δ/2)
            elseif k == N
                P[j, k] = 1 - cdf(Normal(μ_cond, σ), z_grid[k] - Δ/2)
            else
                P[j, k] = cdf(Normal(μ_cond, σ), z_grid[k] + Δ/2) -
                          cdf(Normal(μ_cond, σ), z_grid[k] - Δ/2)
            end
        end
    end

    return z_grid, P
end
