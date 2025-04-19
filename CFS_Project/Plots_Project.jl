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
serialize("CFS_Project/20250419_KV_T.jls", Vector_α_ϵ[1:TR-2])                  
serialize("CFS_Project/20250419_KV_P.jls", Vector_α_ζ[1:TR-2])         

serialize("CFS_Project/20250419_BHRS_T.jls", Vector_α_ϵ)                  
serialize("CFS_Project/20250419_BHRS_P.jls", Vector_α_ζE)              

# Loading data
KV_α_ϵ      = deserialize("CFS_Project/20250419_KV_T.jls")         
KV_α_ζ      = deserialize("CFS_Project/20250419_KV_P.jls")  
BHRS_α_ϵ    = deserialize("CFS_Project/20250419_BHRS_T.jls")         
BHRS_α_ζ    = deserialize("CFS_Project/20250419_BHRS_P.jls")  

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
savefig("CFS_Project/Figures/Project_Image_05.png") 

# Age profiles of insurance coefficients: Permanent shocks
plot(age_grid_short[2:TR-1], KV_α_ζ,
     label      = "KV",
     lw         = 2,                  
     color      = :black)
plot!(age_grid_short[2:TR-1], BHRS_α_ζ,
     label      = "BHRS",
     lw         = 2,                  
     color      = :blue)
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
savefig("CFS_Project/Figures/Project_Image_06.png") 

