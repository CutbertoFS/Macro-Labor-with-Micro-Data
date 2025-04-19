using Distributions

function tauchen(N::Int, ρ::Float64, σ::Float64; μ::Float64=0.0, m::Float64=3.0, grid::Union{Nothing, Vector{Float64}}=nothing)
    """
    Discretizes an AR(1) process: y' = μ + ρ * y + ε, ε ~ N(0, σ^2)

    Arguments:
        N    : Number of grid points
        ρ    : AR(1) persistence parameter
        σ    : Standard deviation of ε
        μ    : Mean of the process (default = 0)
        m    : Width parameter for grid range (default = 3 standard deviations)
        grid : Optional user-supplied grid vector

    Returns:
        y_grid : Vector of grid points
        P      : Transition matrix (N x N)
    """

    # Use supplied grid or construct default grid
    if isnothing(grid)
        σ_y = σ / sqrt(1 - ρ^2)
        y_min = μ - m * σ_y
        y_max = μ + m * σ_y
        y_grid = range(y_min, y_max, length=N) |> collect
    else
        y_grid = grid
    end

    Δ = y_grid[2] - y_grid[1]  # Step size
    P = zeros(N, N)

    for j in 1:N
        for k in 1:N
            μ_cond = μ + ρ * (y_grid[j] - μ)

            if k == 1
                P[j, k] = cdf(Normal(μ_cond, σ), y_grid[k] + Δ/2)
            elseif k == N
                P[j, k] = 1 - cdf(Normal(μ_cond, σ), y_grid[k] - Δ/2)
            else
                P[j, k] = cdf(Normal(μ_cond, σ), y_grid[k] + Δ/2) -
                          cdf(Normal(μ_cond, σ), y_grid[k] - Δ/2)
            end
        end
    end

    return y_grid, P
end
