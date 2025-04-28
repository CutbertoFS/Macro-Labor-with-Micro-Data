#= ################################################################################################## 

    Econ 810: Spring 2025 Advanced Macroeconomics 
    Final Project: Unemployment risk in a Life-cycle Bewely economy

    Last Edit:  April 19, 2025
    Authors:    Cutberto Frias Sarraf

=# ##################################################################################################

using Parameters, Plots, Random, LinearAlgebra, Statistics, LaTeXStrings, Distributions, Serialization

#= ################################################################################################## 
    
=# ##################################################################################################

# Saving data

# KV, NOT GREAT RESULTS: σ_ζ::Float64  = sqrt(0.095)
serialize("CFS_Project/Figures_PPT_New/20250427_KV_T.jls", Vector_α_ϵ[1:TR-2])                  
serialize("CFS_Project/Figures_PPT_New/20250427_KV_P.jls", Vector_α_ζ[1:TR-2])         

# KV, NICE RESULTS: σ_ζ::Float64       = sqrt(0.01)
serialize("CFS_Project/Figures_PPT_New/20250427_KV_T_Alt.jls", Vector_α_ϵ[1:TR-2])                  
serialize("CFS_Project/Figures_PPT_New/20250427_KV_P_Alt.jls", Vector_α_ζ[1:TR-2])  

# BHRS
serialize("CFS_Project/Figures_PPT_New/20250427_BHRS_T.jls", Vector_α_ϵ)                  
serialize("CFS_Project/Figures_PPT_New/20250427_BHRS_P.jls", Vector_α_ζE)     


# Loading data: Option 1
KV_α_ϵ      = deserialize("CFS_Project/Figures_PPT_New/20250427_KV_T_Alt.jls")         
KV_α_ζ      = deserialize("CFS_Project/Figures_PPT_New/20250427_KV_P_Alt.jls")  
BHRS_α_ϵ    = deserialize("CFS_Project/Figures_PPT_New/20250427_BHRS_T.jls")         
BHRS_α_ζ    = deserialize("CFS_Project/Figures_PPT_New/20250427_BHRS_P.jls")  

# Loading data: Option 2
KV_α_ϵ      = deserialize("CFS_Project/Figures_PPT_New/20250427_KV_T.jls")         
KV_α_ζ      = deserialize("CFS_Project/Figures_PPT_New/20250427_KV_P.jls")  
BHRS_α_ϵ    = deserialize("CFS_Project/Figures_PPT_New/20250427_BHRS_T.jls")         
BHRS_α_ζ    = deserialize("CFS_Project/Figures_PPT_New/20250427_BHRS_P.jls")  

#= ################################################################################################## 
    Plots
=# ##################################################################################################

age_grid_short = collect(Int, range(25, 59, length=35)) 

# Age profiles of insurance coefficients: Transitory shocks
plot(age_grid_short[2:TR-1], KV_α_ϵ,
     label      = "KV",
     lw         = 2,                  
     color      = :black)
plot!(age_grid_short[2:TR-1], BHRS_α_ϵ,
     label      = "BHRS",
     lw         = 2,                  
     color      = :blue)
# title!(L"\mathsf{Insurance\ Coefficients:\ Transitory\ Shocks}")
xlabel!("Age")
ylabel!(L"\mathsf{Insurance\ Coefficient}\ \phi^{\varepsilon}")
plot!(legend = :bottomright,             
      xtickfont  = font(12),
      ytickfont  = font(12),
      guidefont  = font(14),
      legendfont = font(14),
      titlefont  = font(12), 
      size       = (700, 500))     
savefig("CFS_Project/Figures_PPT_New/Transitory_Both_Alt.png") 

# Age profiles of insurance coefficients: Permanent shocks
plot(age_grid_short[2:TR-1], KV_α_ζ,
     label      = "KV",
     lw         = 2,                  
     color      = :black)
plot!(age_grid_short[2:TR-1], BHRS_α_ζ,
     label      = "BHRS",
     lw         = 2,                  
     color      = :blue)
# title!(L"\mathsf{Insurance\ Coefficients:\ Persistent\ Shocks}")
xlabel!("Age")
ylabel!(L"\mathsf{Insurance\ Coefficient}\ \phi^{\eta}")
plot!(legend = :topleft,             
      xtickfont  = font(12),
      ytickfont  = font(12),
      guidefont  = font(14),
      legendfont = font(14),
      titlefont  = font(12), 
      size       = (700, 500))   
savefig("CFS_Project/Figures_PPT_New/Persistent_Both_Alt.png") 

#= ################################################################################################## 
    
=# ##################################################################################################