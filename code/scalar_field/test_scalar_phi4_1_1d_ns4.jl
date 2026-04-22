# %%
using Statistics: mean, std
using Printf: @sprintf, @printf
using Printf: @sprintf as @s
using CairoMakie: Figure, Axis, scatter!, lines!, save
using ProgressMeter: @showprogress, Progress, next!
using LaTeXStrings: latexstring, @L_str
import HDF5
import LinearAlgebra.BLAS

# %%
BLAS.set_num_threads(1)

# %%
const m = 0.5 # in units of a
const a_0 = 0.05 # in units of a
const λ = 16.0

const Nt = 500
const Ns = 4 # number of lattice sites spatially

const eps = 0.5 / sqrt(Ns) # fluctuation for each update

const batch_size = 1000
const n_batches = 10
const N_sets = 8
# const N_cf = 40000
const N_cor = 100
const N_thermal = 100 * N_cor

const plateau_threshold = 0.1
const subtract_ratio = 1.

output_path = "output"
fig_path    = "figs"
mkpath(output_path)
mkpath(fig_path)
name = "phi4_norm_ns$(Ns)"
data_name = joinpath(output_path, name * ".h5")


# %%
@inline sumr(A; dims) = dropdims(sum(A, dims=dims), dims=dims)
@inline meanr(A; dims) = dropdims(mean(A, dims=dims), dims=dims)
@inline maximumr(A; dims) = dropdims(maximum(A, dims=dims), dims=dims)

@inline pow4(x) = x * x * x * x


# %%
function free_energy_per_mode(p, m=m)
    return sqrt(m^2 + (2 * sin(p/2))^2) / 2
end

function free_ground_state_energy(m=m, Ns=Ns)
    if Ns % 2 != 0
        throw(ArgumentError("Only even Ns is supported"))
    end
    # mode_list = LinRange(-Ns // 2, Ns // 2, Ns) * 2π / Ns
    mode_list = collect(-Ns ÷ 2 : Ns ÷ 2 - 1) * 2π / Ns
    return sum(free_energy_per_mode.(mode_list, m))
end

# %%
struct MCWorkspace
    lat::Array{Float64, 2}
    lat_temp::Array{Float64, 2}
    fluctuation::Array{Float64, 2}
    S_old::Vector{Float64}
    S_new::Vector{Float64}
    S_draft::Array{Float64, 2}
    success_flags::Vector{Float64}

    function MCWorkspace(nt::Integer, ns::Integer)
        new(zeros(nt, ns), zeros(nt, ns), zeros(nt, ns),
            zeros(nt), zeros(nt), zeros(nt, ns), zeros(nt))
    end
end

@kwdef struct MCSingleResult
    energies::Vector{Float64}
    phi2max_list::Vector{Float64}
    phi2_mean::Float64
    pi2_mean::Float64
    accept_rate::Float64
    corrs::Vector{Float64}
end

struct MCBinningResult
    bin_sizes::Vector{Float64}
    means::Vector{Float64}
    errs::Vector{Float64}
end

# %% [markdown]
# Hamiltonian is 
# 
# $$
# H =\sum_{x} \frac{1}{2} \pi^2 + \frac{1}{2} m^2 \phi^2 - \frac{1}{2} \phi \partial^2 \phi + \frac{1}{4!} \lambda * \phi^4
# $$

# %% [markdown]
# Action is
# 
# $$
# -S = \sum_{t, x} \left( -\frac{a}{a_0}\frac{(\phi'-\phi)^2}{2}-\frac{a_0}{a}\frac{m^2 \phi^2}{2}+\frac{a_0}{a} \frac{1}{2}\phi \partial_x^2 \phi -\frac{a_0}{a}\frac{\lambda \phi^4}{4!}\right)
# $$

# %%
function S_local_lat(lat)
    # lat shape (Nt, Ns)
    result = zeros(Nt, Ns)

    @inbounds for i in 1:Nt
        i_prev = mod1(i-1, Nt)
        i_next = mod1(i+1, Nt)
        @simd for j in 1:Ns
            j_prev = mod1(j-1, Ns)
            j_next = mod1(j+1, Ns)

            # compute not the action itself, but the "part" that will change
            result[i, j] = a_0 * (m * lat[i, j])^2 / 2 # mass term
            result[i, j] += a_0 * lat[i, j] * (lat[i, j] - lat[i, j_prev] - lat[i, j_next]) # spatial term
            result[i, j] += 1 / a_0 * lat[i, j] * (lat[i, j] - lat[i_prev, j] - lat[i_next, j]) # temporal term
        end
    end

    return result
end

function S_lat_tloc!(lat::Array{Float64, 2}, s_lat_t::Vector{Float64}, s_draft::Array{Float64, 2}; m = m,
        a_0 = a_0, λ = λ)

    s_draft .= 0. # initialize to zero
    s_lat_t .= 0.

    inv_a_0 = 1 / a_0  # precompute division
    m_sq_half = a_0 * m^2 / 2  # precompute mass coefficient
    λ_24 = a_0 * λ / 24  # precompute λ coefficient

    @inbounds for j in 1:Ns
        # j_prev = mod1(j-1, Ns)
        j_next = mod1(j+1, Ns)

        @simd for i in 1:Nt
            i_prev = mod1(i-1, Nt)
            i_next = mod1(i+1, Nt)

            phi = lat[i, j]


            s_val = m_sq_half * phi^2 # mass term
            s_val += λ_24 * pow4(phi) # phi^4 term 
            s_val += a_0 * (phi - lat[i, j_next])^2 / 2 # spatial term
            s_val += inv_a_0 * phi * (phi- lat[i_prev, j] - lat[i_next, j]) # temporal term

            s_draft[i, j] = s_val
            s_lat_t[i] += s_val 
        end
    end

    nothing
end

# %%
const odd_time = [Float64(i % 2) for i in 1:Nt]
const even_time = 1 .- odd_time;

function update!(wc::MCWorkspace, get_S_tloc! = S_lat_tloc!)
    
    wc.fluctuation .= rand.() .* 2 .* eps .- eps
    # fluctuation = rand(Nt, Ns, Ns) .* 2 .* eps .- eps

    success_num = 0

    # now, let me do it alternatively between even layers and odd layers of time slice
    # S_old_odd  = get_S_lat(lat)
    # S_new_odd  = get_S_lat(lat .+ fluctuation .* odd_time)
    # # Now doing odd sites
    get_S_tloc!(wc.lat, wc.S_old, wc.S_draft)
    wc.lat_temp .= wc.lat .+ wc.fluctuation .* odd_time
    get_S_tloc!(wc.lat_temp, wc.S_new, wc.S_draft)
    # dS_odd     = S_new_odd .- S_old_odd
    # odd_success = Float64.(exp.(-dS_odd) .> rand(Nt)) .* odd_time
    wc.success_flags .= Float64.(exp.(-wc.S_new .+ wc.S_old) .> rand.()) .* odd_time
    wc.lat .+= wc.fluctuation .* wc.success_flags
    success_num += Int32(sum(wc.success_flags))

    ## Now do even sites
    get_S_tloc!(wc.lat, wc.S_old, wc.S_draft)
    wc.lat_temp .= wc.lat .+ wc.fluctuation .* even_time
    get_S_tloc!(wc.lat_temp, wc.S_new, wc.S_draft)
    wc.success_flags .= Float64.(exp.(-wc.S_new .+ wc.S_old) .> rand.()) .* even_time
    wc.lat .+= wc.fluctuation .* wc.success_flags
    success_num += Int32(sum(wc.success_flags))

    return success_num
end

# %%
function phi_term(confs; m = m, λ = λ)
    # confs (N_cf, Nt, Ns)
    mass_term_site = 0.5 .* m^2 .* confs.^2
    momentum_term_site = 0.5 .* (confs .- circshift(confs, (0, 0, 1))).^2
    λ_term_site = λ .* confs.^4  ./ 24
    
    phi_term_for_a_time = sumr(mass_term_site .+ momentum_term_site .+ λ_term_site, dims=3) # (N_cf, Nt)
    return meanr(phi_term_for_a_time, dims=2) # (N_cf)
end

function pi_term(confs; a_0 = a_0)
    # confs (N_cf, Nt, Ns)
    delta_phi = confs .- circshift(confs, (0, 1, 0))
    kin_term_site = 0.5 * (1/a_0 .- 1/a_0^2 .* delta_phi.^2)
    pi_term_for_a_time = sumr(kin_term_site, dims=3) # (N_cf, Nt)
    return meanr(pi_term_for_a_time, dims=2) # (N_cf)
end

function estimate_pi2(confs; a_0 = a_0)
    return 2 * mean(pi_term(confs; a_0))
end

function Hamiltonian(confs; m = m, λ = λ, a_0=a_0)
    return phi_term(confs; m, λ) .+ pi_term(confs; a_0)
end

#=
function get_correlator(confs, d)
    return mean(confs .* circshift(confs, (0, d, 0)))
end
=#

function get_correlator(confs, d)
    s1, s2, s3 = size(confs)
    total = 0.0
    
    @inbounds for k in 1:s3
        for j in 1:s2
            j_shifted = mod1(j + d, s2)  # periodic boundary
            for i in 1:s1
                total += confs[i, j, k] * confs[i, j_shifted, k]
            end
        end
    end
    
    return total / length(confs)
end

# %%
function Metropolis_single(S_tloc! = S_lat_tloc! , get_Hamiltonian = Hamiltonian, get_pi2 = estimate_pi2 ; Nt::Int = Nt, Ns::Int = Ns,
        n_batches::Int = n_batches, batch_size::Int = batch_size, p::Progress)
#     lat = zeros(Nt, Ns)
    paths = zeros(batch_size, Nt, Ns)

    energies = Array{Float64}(undef, batch_size * n_batches)
    phi2max_list = Array{Float64}(undef, batch_size * n_batches)
    phi2_mean_ls = Array{Float64}(undef, n_batches)
    pi2_mean_ls = Array{Float64}(undef, n_batches)
    corrs_vals_ls = Array{Float64}(undef, Nt, n_batches)

    tot_updates = 0
    acc_updates = 0

    wc = MCWorkspace(Nt, Ns)

    for j in 1:N_thermal
        sn = update!(wc, S_tloc!)
        tot_updates += Nt
        acc_updates += sn
    end

    for n in 1:n_batches
        for i in 1:batch_size
            for j in 1:N_cor
                sn = update!(wc, S_tloc!)
                tot_updates += Nt
                acc_updates += sn
            end
            paths[i, :, :] = wc.lat

            next!(p) # notify the progress meter
        end

        i_s = (n - 1) * batch_size + 1
        i_e = i_s + batch_size - 1
        energies[i_s:i_e] = get_Hamiltonian(paths)
        phi2max_list[i_s:i_e] = meanr(dropdims(maximum(paths.^2, dims=(3)), dims=(3)), dims=2)
        phi2_mean_ls[n] = mean(paths.^2)
        pi2_mean_ls[n] = get_pi2(paths)
        corrs_vals_ls[:, n] = [get_correlator(paths, t) for t in 0:Nt-1]
    end

    accept_rate = acc_updates/tot_updates
    phi2_mean = mean(phi2_mean_ls)
    pi2_mean = mean(pi2_mean_ls)
    corrs = meanr(corrs_vals_ls, dims = 2)
    
    result = MCSingleResult(;energies, phi2max_list, phi2_mean, pi2_mean, accept_rate, corrs)
    

    return result
end

function Metropolis_parallel(S_tloc! = S_lat_tloc! , get_Hamiltonian = Hamiltonian, get_pi2 = estimate_pi2; debug=false, Nt::Int = Nt, Ns::Int = Ns, 
    n_batches::Int = n_batches, batch_size::Int = batch_size, N_sets::Int = N_sets)

    N_cf_s = n_batches * batch_size
    N_cf_tot = N_cf_s * N_sets
    p = Progress(N_cf_tot; dt=1.0)

    energies = Array{Float64}(undef, N_cf_tot)
    phi2max_list = Array{Float64}(undef, N_cf_tot)
    phi2_mean_ls = Array{Float64}(undef, N_sets)
    pi2_mean_ls = Array{Float64}(undef, N_sets)
    accept_rate_ls = Array{Float64}(undef, N_sets)
    corrs_vals_ls = Array{Float64}(undef, Nt, N_sets)

    Threads.@threads for i in 1:N_sets
        i_start = (i-1) * N_cf_s + 1
        i_end = i_start + N_cf_s - 1

        result_i = Metropolis_single(S_tloc! , get_Hamiltonian, get_pi2; Nt, Ns, n_batches, batch_size, p)

        energies[i_start:i_end] = result_i.energies
        phi2max_list[i_start:i_end] = result_i.phi2max_list
        phi2_mean_ls[i] = result_i.phi2_mean
        pi2_mean_ls[i] = result_i.pi2_mean
        accept_rate_ls[i] = result_i.accept_rate
        corrs_vals_ls[:, i] = result_i.corrs
    end

    phi2_mean = mean(phi2_mean_ls)
    pi2_mean  = mean(pi2_mean_ls)
    accept_rate = mean(accept_rate_ls)
    corrs = meanr(corrs_vals_ls, dims = 2)

    if debug
       println("Accept rate is $(accept_rate)")
    end

    result = MCSingleResult(;energies, phi2max_list, phi2_mean, pi2_mean, accept_rate, corrs)
    return result
end

# %%
"""
    binning_analysis(data::Vector{Float64}, max_bin_size::Int=0)

Perform binning analysis on time series data to estimate statistical errors
accounting for autocorrelations. Returns bin sizes, means, and standard errors.

# Arguments
- `data`: Vector of measurements (e.g., observables from MC configurations)
- `max_bin_size`: Maximum bin size to consider (default: length(data)÷4)

# Returns
- `bin_sizes`: Vector of bin sizes used
- `means`: Mean value for each binning level
- `errors`: Standard error for each binning level
"""
function binning_analysis(data::Vector{Float64}, max_bin_size::Int=0)
    n = length(data)
    
    if max_bin_size == 0
        max_bin_size = n ÷ 8  # Default: use up to n/8
    end
    
    bin_sizes = Int[]
    means = Float64[]
    errors = Float64[]
    
    # Try different bin sizes (powers of 2 for efficiency)
    bin_size = 1
    while bin_size <= max_bin_size
        n_bins = n ÷ bin_size
        
        if n_bins < 2  # Need at least 2 bins for error estimate
            break
        end
        
        # Create bins by averaging
        bin_means = zeros(n_bins)
        for i in 1:n_bins
            start_idx = (i-1) * bin_size + 1
            end_idx = i * bin_size
            bin_means[i] = mean(data[start_idx:end_idx])
        end
        
        # Calculate statistics from binned data
        bin_mean = mean(bin_means)
        bin_error = std(bin_means) / sqrt(n_bins)
        
        push!(bin_sizes, bin_size)
        push!(means, bin_mean)
        push!(errors, bin_error)
        
        bin_size *= 2
    end
    
    return bin_sizes, means, errors
end


"""
    print_binning_results(bin_sizes, means, errors)

Pretty print the results of binning analysis.
"""
function print_binning_results(bin_sizes, means, errors)
    println("\nBinning Analysis Results:")
    println("="^60)
    println(@sprintf("%12s %15s %15s", "Bin Size", "Mean", "Std Error"))
    println("-"^60)
    
    for (bs, m, e) in zip(bin_sizes, means, errors)
        println(@sprintf("%12d %15.8f %15.8f", bs, m, e))
    end

end



"""
Find the plateau of the binning analysis.

The function will find the change in error, and see whether the relative difference is below threshold.
    Then, the function will find the longest sequence of "plateau"
    
The function returns two integers: i and d
    The "plateau" part of the arrays is errors[i:i+d]
"""
function find_binning_plateau_index(errors; threshold = 0.1)
    i_start = firstindex(errors)
    i_end = lastindex(errors)

    i_current = i_start
    d_current = 0
    i_best = i_start
    d_best = 0

    for i in i_start:i_end-1
        rel_err = abs( (errors[i+1]-errors[i]) /errors[i])
        if rel_err >= threshold
            if i - i_current >= d_best
                i_best = i_current
                d_best = i - i_current
            end
            i_current = i + 1
        end
    end

    if i_end - i_current >= d_best
        i_best = i_current
        d_best = i_end - i_current
    end

    return i_best, d_best
end

"""
Do a conservative estimate of the error from binning analysis.

Find the max error between the start of the plateau and the end of the whole array.
"""
function find_err_from_binning(errors; threshold = 0.1, minimum_plateau_len = 2, print_plateau = false)
    plateau_start, plateau_interval_diff = find_binning_plateau_index(errors; threshold)
    errors_plateau = view(errors, plateau_start:(plateau_start+plateau_interval_diff))

    if print_plateau
        println(errors_plateau)
    end

    if plateau_interval_diff < minimum_plateau_len
        println("Warning: Could not find a plateau with width $(minimum_plateau_len)")
        println("Max plateau width is $(plateau_interval_diff)")
    end

    error_final = maximum(errors[plateau_start:end])
    return error_final
end

# %%
@time MC_result = Metropolis_parallel(;debug=true);

# %%
gs_unmodified_energies = MC_result.energies
gs_unmodified_energies_mean  = mean(gs_unmodified_energies)
gs_unmodified_energies_std   = std(gs_unmodified_energies)
# gs_unmodified_energies_delta = gs_unmodified_energies_std ./ sqrt(N_cf)
println(gs_unmodified_energies_mean)
println(gs_unmodified_energies_std)

# %%
println(free_ground_state_energy())

# %%
bin_sizes, means, errors = binning_analysis(gs_unmodified_energies)
print_binning_results(bin_sizes, means, errors)
gs_unmodified_energies_binning_err = find_err_from_binning(errors; threshold=plateau_threshold)


# %%
println(@sprintf "Ground state energy from Monte Carlo is %.5f±%.5f" gs_unmodified_energies_mean gs_unmodified_energies_binning_err)

# %%
# println("Exact ground state energy is $(free_ground_state_energy())")

# %%
MC_result.phi2_mean

# %%
MC_result.pi2_mean




# %%
phi2max_mean = mean(MC_result.phi2max_list)

# %%
println("-------------------------------")

# %%
binning_unmodified = MCBinningResult(bin_sizes, means, errors)

# %%
gs_unmodified_energies_delta = gs_unmodified_energies_binning_err

# %%
println("For the unmodified vacuum, we have:")
println("Average <φ^2> is $(mean(MC_result.phi2_mean))")
println("Average <π^2> is $(mean(MC_result.pi2_mean))")

# %%
function plot_correlator(d_ls, corr_ls)
    r_list = circshift(corr_ls, -1) ./ corr_ls
    fig = Figure(size=(1200, 400))
    
    ax1 = Axis(fig[1, 1], xlabel="Distance", ylabel="Correlation")
    lines!(ax1, d_ls, corr_ls)
    
    ax2 = Axis(fig[1, 2], xlabel="Distance", ylabel=L"Correlator ratio $\frac{A_{t+1}}{A_t}$")
    lines!(ax2, d_ls, r_list)
    
    ax3 = Axis(fig[1, 3], xlabel="Distance", ylabel="Correlation", yscale=log10)
    lines!(ax3, d_ls, abs.(corr_ls))

    return fig
end




# %%
f = plot_correlator(0:Nt-1, MC_result.corrs)
save_name = joinpath(fig_path, "$(name)_corr.pdf")
save(save_name, f)

# %%


function S_lat_tloc_minusPhi2max!(lat::Array{Float64, 2}, s_lat_t::Vector{Float64}, s_draft::Array{Float64, 2}; m = m,
        a_0 = a_0, λ = λ, subtract_ratio=subtract_ratio)

    s_draft .= 0. # initialize to zero
    s_lat_t .= 0.

    inv_a_0 = 1 / a_0  # precompute division
    m_sq_half = a_0 * m ^2 / 2  # precompute mass coefficient
    λ_24 = a_0 * λ / 24  # precompute λ coefficient

    @inbounds for j in 1:Ns
        # j_prev = mod1(j-1, Ns)
        j_next = mod1(j+1, Ns)

        @simd for i in 1:Nt
            i_prev = mod1(i-1, Nt)
            i_next = mod1(i+1, Nt)

            phi = lat[i, j]


            s_val = m_sq_half * phi^2 # mass term
            s_val += λ_24 * pow4(phi) # phi^4 term 
            s_val += a_0 * (phi - lat[i, j_next])^2 / 2 # spatial term
            s_val += inv_a_0 * phi * (phi- lat[i_prev, j] - lat[i_next, j]) # temporal term

            s_draft[i, j] = s_val
            s_lat_t[i] += s_val 
        end
    end

    s_lat_t .-= (subtract_ratio * a_0 * m^2 / 2) .* maximumr(lat .^ 2, dims=2)

    nothing
end

function phi_term_minusPhi2max(confs; m = m, λ = λ, subtract_ratio=subtract_ratio)
    # confs (N_cf, Nt, Ns)
    mass_term_site = 0.5 .* m^2 .* confs.^2
    momentum_term_site = 0.5 .* (confs .- circshift(confs, (0, 0, 1))).^2
    λ_term_site = λ .* confs.^4  ./ 24
    
    phi_term_for_a_time = sumr(mass_term_site .+ momentum_term_site .+ λ_term_site, dims=3) # (N_cf, Nt)
    phi_term_for_a_time .-= (subtract_ratio / 2 * m^2) .* maximumr(confs.^2, dims=3)
    return meanr(phi_term_for_a_time, dims=2) # (N_cf)
end

function Hamiltonian_minusPhi2max(confs; m = m, λ = λ, a_0=a_0, subtract_ratio=subtract_ratio)
    return phi_term_minusPhi2max(confs; m, λ, subtract_ratio) .+ pi_term(confs; a_0)
end


# %%
S_lat_tmp!(lat::Array{Float64, 2}, s_lat_t::Vector{Float64}, s_draft::Array{Float64, 2}) = 
let subtract_ratio = subtract_ratio
    S_lat_tloc_minusPhi2max!(lat, s_lat_t, s_draft; subtract_ratio)
end

get_Hamiltonian_tmp(confs::Array{Float64, 3}) = 
let subtract_ratio = subtract_ratio
    Hamiltonian_minusPhi2max(confs; subtract_ratio)
end

println("\n\n---------------------φ2--------------------")


@time MC_phi2_result = Metropolis_parallel(S_lat_tmp!, get_Hamiltonian_tmp; debug=true)

gs_minusPhi2_energies = MC_phi2_result.energies
gs_minusPhi2_energies_mean  = mean(gs_minusPhi2_energies)
gs_minusPhi2_energies_std   = std(gs_minusPhi2_energies)
phi2_E_bin_sizes, phi2_E_means, phi2_E_errors = binning_analysis(gs_minusPhi2_energies)
gs_minusPhi2_energies_delta = find_err_from_binning(phi2_E_errors; threshold=plateau_threshold)
binning_phi2_energies = MCBinningResult(phi2_E_bin_sizes, phi2_E_means, phi2_E_errors)
println("Average energy is $gs_minusPhi2_energies_mean")
println("Std of energy is $gs_minusPhi2_energies_std")
println("Naive uncertainty of energy: $gs_minusPhi2_energies_delta")


energy_diff = gs_unmodified_energies_mean - gs_minusPhi2_energies_mean
energy_diff_delta = hypot(gs_unmodified_energies_delta, gs_minusPhi2_energies_delta)
phi2bound = energy_diff * 2  / subtract_ratio / m^2
phi2bound_sqrt = phi2bound >=0 ? sqrt(phi2bound) : NaN
phi2bound_with_delta  = (energy_diff + energy_diff_delta) * 2 / subtract_ratio / m^2
phi2bound_with_3delta = (energy_diff + 3 * energy_diff_delta) * 2 / subtract_ratio / m^2
phi2bound_sqrt_with_delta = phi2bound_with_delta >=0 ? sqrt(phi2bound_with_delta) : NaN
phi2bound_sqrt_with_3delta = phi2bound_with_3delta >=0 ? sqrt(phi2bound_with_3delta) : NaN

phi2max_phi2_mean = mean(MC_phi2_result.phi2max_list)
phi2_max_bin_sizes, phi2_max_means, phi2_max_errors = binning_analysis(MC_phi2_result.phi2max_list)
binning_phi2_phi2max = MCBinningResult(phi2_max_bin_sizes, phi2_max_means, phi2_max_errors)
phi2max_phi2_delta = find_err_from_binning(binning_phi2_phi2max.errs; threshold=plateau_threshold)

println("Binning result for the energy")
print_binning_results(phi2_E_bin_sizes, phi2_E_means, phi2_E_errors)
println("\n\nBinning result for φ_max^2")
print_binning_results(phi2_max_bin_sizes, phi2_max_means, phi2_max_errors)

println("Energy difference is about $energy_diff ± $energy_diff_delta")
println("This means <φ^2> is bounded by $phi2bound, whose square root is $phi2bound_sqrt")
println("Or with one delta, <φ^2> is bounded by $(phi2bound_with_delta), whose square root is $(phi2bound_sqrt_with_delta)")
println("Or with three deltas, <φ^2> is bounded by $(phi2bound_with_3delta), whose square root is $(phi2bound_sqrt_with_3delta)")

phi2_ref = mean(MC_result.phi2_mean)
println("\nAs a reference, for the unmodified Hamiltonian <φ^2> is $(phi2_ref)")
println("Also, for the unmodified vacuum, <φ_max^2> is $(phi2max_mean), the sqrt of which is $(sqrt(phi2max_mean))")
println("Also, for the modified vacuum, <φ_max^2> is $(phi2max_phi2_mean), the sqrt of which is $(sqrt(phi2max_phi2_mean))")
println("To be more exact, it's $(phi2max_phi2_mean)±$(phi2max_phi2_delta)")
println("\n\n")

f = plot_correlator(collect(0:Nt-1), MC_phi2_result.corrs)

save_name = joinpath(fig_path, "$(name)_phi2_corr.pdf")
save(save_name, f)

# %%


HDF5.h5open(data_name, "w") do file
    HDF5.create_group(file, "parameters")
    file["parameters"]["Ns"]  = Ns
    file["parameters"]["Nt"]  = Nt
    file["parameters"]["a_0"] = a_0
    file["parameters"]["m"]   = m

    HDF5.create_group(file, "unmodified")
    file["unmodified"]["energy_mean"] = gs_unmodified_energies_mean
    file["unmodified"]["energy_delta"] = gs_unmodified_energies_binning_err
    file["unmodified"]["phi2max"] = mean(MC_result.phi2max_list)
    file["unmodified"]["phi2_mean"] = MC_result.phi2_mean
    file["unmodified"]["pi2_mean"] = MC_result.pi2_mean
    file["unmodified"]["corrs"] = MC_result.corrs
    HDF5.create_group(file["unmodified"], "binning_energy")
    file["unmodified/binning_energy"]["bin_sizes"] = binning_unmodified.bin_sizes
    file["unmodified/binning_energy"]["means"] = binning_unmodified.means
    file["unmodified/binning_energy"]["errs"] = binning_unmodified.errs


    HDF5.create_group(file, "phi2")
    file["phi2"]["description"] = "Bounding all φ2 once by bounding (φ_max)^2"
    file["phi2"]["energy_mean"] = gs_minusPhi2_energies_mean
    file["phi2"]["energy_delta"] = gs_minusPhi2_energies_delta
    file["phi2"]["phi2max_mean"] = phi2max_phi2_mean
    file["phi2"]["phi2max_delta"] = phi2max_phi2_delta
    file["phi2"]["corrs"] = MC_phi2_result.corrs

    HDF5.create_group(file["phi2"], "binning_energy")
    HDF5.create_group(file["phi2"], "binning_phi2max")
    file["phi2/binning_energy"]["bin_sizes"] = binning_phi2_energies.bin_sizes
    file["phi2/binning_energy"]["means"] = binning_phi2_energies.means
    file["phi2/binning_energy"]["errs"] = binning_phi2_energies.errs
    file["phi2/binning_phi2max"]["bin_sizes"] = binning_phi2_phi2max.bin_sizes
    file["phi2/binning_phi2max"]["means"] = binning_phi2_phi2max.means
    file["phi2/binning_phi2max"]["errs"] = binning_phi2_phi2max.errs
end
