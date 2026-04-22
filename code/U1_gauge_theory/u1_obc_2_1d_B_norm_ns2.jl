# %% [markdown]
# The bug in test0 has been corrected.

# note in 1_9_8: also fixed an error in 1_9_7_1 (bounding R2)

# %%
using Random:rand
using Printf: @sprintf, @printf
using Printf: @sprintf as @s
using LinearAlgebra: eigen, inv, pinv, diag, Diagonal, mul!
import LinearAlgebra.BLAS
using Statistics: mean, std
using CairoMakie: Figure, Axis, scatter!, lines!, save
using LaTeXStrings: latexstring, @L_str
using ProgressMeter: @showprogress, Progress, next!
# import Logging: @warn
# using StyledStrings: @styled_str
import HDF5
# import JLD2

# %%
BLAS.set_num_threads(1)

# %%
# every quantity is given in terms of a


const a_0 = 0.05 # in units of a
const g   = 1.0 # dimensionless g, in units of a^(-1/2)

const Nt = 200
const Ns = 2 # number of lattice sites spatially per dimension

const subtract_ratio = 1/2

const eps = 2 / Ns^(3/2) * g # fluctuation for each update

const N_sets = 8
# const N_cf_s    = 5000
const batch_size = 500
const n_batches = 10
const N_cf_tot  = N_sets * n_batches * batch_size
const N_cor = 25 * Ns
const N_thermal = 50 * Ns * N_cor

const plateau_threshold = 0.1

plaquettes_range = 1:1 # for now let's just consider one range
output_path = "output"
fig_path    = "figs"
mkpath(output_path)
mkpath(fig_path)
name = "obc_norm_ns$(Ns)"
data_name = joinpath(output_path, name * ".h5")


# %%
# Let's plot the correlator at some specific sites just to check the validity of MC
if Ns >= 4
    corr_watch_ls = [(1, 1), (Ns ÷ 2, Ns ÷ 2)]
else
    corr_watch_ls = [(1, 1)]
end

# %%
# helper functions
@inline sumr(A; dims) = dropdims(sum(A, dims=dims), dims=dims)
@inline meanr(A; dims) = dropdims(mean(A, dims=dims), dims=dims)

# %% [markdown]
# For the convention, let's say we have $N\times N$ plaquettes. 
# 
# For each plaquette, theres a coordinate $(n_x, n_y)$, where both $n_x$ and $n_y$ go from $1$ to $N$. (Or in python's convention, from $0$ to $N-1$).
# 
# We will flatten the plaquette variables into a vector, e.g., $R_i$, where $i$ go from $1$ to $N^2$ (or $0$ to $N^2-1$). Let's use the notation that $i = n_x + (n_y - 1) * N$ (or in python's notation: $i = n_x + n_y * N$).

# %% [markdown]
# In the OBC, we have $(N+1)\times (N+1)$ vertices.
# 
# In terms of edges, we have $2 N^2+2N$ links. For each link $E_{n_x, n_y, x}$ or $E_{n_x, n_y, y}$, we have three indices, where $n_x$ and $n_y$ indicate the position of the vertice and go from $1$ to $N+1$ (or $0$ to $N$ in python), and $x$ or $y$ indicates whether the gauge link is to the $+\hat{e_x}$ direction or to the $+\hat{e_y}$ direction from the point of view of the $(n_x, n_y)$ point.
# 
# In the code, I will use a tuple $(n_x, n_y, p)$ to denote the position of the link. $p$ will take the value $0$ or $1$, indicating $x$ or $y$. 
# 
# We will flatten $E_{n_x, n_y, x}$ into $E_i$. Here, $i$ go from $1$ to $2N^2+2N$ (or $0$ to $2N^2+2N - 1$. The convention is a bit complex. For the first $2N^2$ indices, we use the convention that $i = 2(n_x +(n_y - 1) * N) -1 + p_x$, where $p_x=0$ for $x$ and $p_x=1$ for $y$ (or in python language, we have $i = 2(n_x + n_y * N) + p_x~~~~$    ).  For the next $N$ indices, we have $E_i = E_{N, n_y, y}$ with $i = 2N^2 + n_y$ (this is true for both julia and python convention). For the last $N$ indices, we have $E_i = E_{n_x, N, x}$ with $i = 2N^2 + N + n_x$ (this is true for both julia and python convention).

# %%
function plaquette_i_to_coord(i, N=Ns)
    if i <= 0 || i > N^2
        throw(ArgumentError("i is out of range. Right now, i is $i"))
    end
    n_x = mod1(i, N)
    n_y = (i - 1) ÷ N + 1
    return n_x, n_y
end

function plaquette_coord_to_i(n_x, n_y, N=Ns)
    i = n_x + (n_y - 1) * N
    return i
end

function link_coord_to_i(n_x, n_y, p, N=Ns)
    if n_x <= N && n_y <= N
        return 2 * (n_x + (n_y-1) * N) - 1 + p
    elseif n_x == N+1 && n_y <= N
        if p == 1
            return 2 * N^2 + n_y
        else
            throw(ArgumentError("Not valid"))
        end
    elseif n_x <= N && n_y == N + 1
        if p == 0
            return 2 * N^2 + N + n_x
        else
            throw(ArgumentError("Not valid"))
        end
    else
        throw(ArgumentError("Index out of range. Here, n_x is $(n_x) and n_y is $(n_y)"))
    end
end

function link_i_to_coord(i, N=Ns)
    if 1 <= i <= 2 * N^2
        p = (i + 1) % 2
        tmp = div(i+1-p, 2)
        n_x = tmp % N
        n_y = div(tmp - 1, N) + 1
        return n_x, n_y, p
    elseif i <= 2 * N^2 + N
        n_x = N + 1
        n_y = i - 2 * N^2
        p = 1
        return n_x, n_y, p
    elseif i < 2 * N^2 + 2 * N
        n_x = i - 2 * N^2 - N
        n_y = N + 1
        p = 0
        return n_x, n_y, p
    else
        throw(ArgumentError("Index out of range"))
    end
end

# %%
plaquette_i_to_coord(1, 2)

# %%
link_coord_to_i(1, 1, 0, 2)

# %%
function get_mat_RtoE(N)
    n_col = N^2
    n_row = 2 * N^2 + 2 * N

    mat = zeros(n_row, n_col)

    for j in 1:n_col
        nx, ny = plaquette_i_to_coord(j, N)
        
        # edges
        i1 = link_coord_to_i(nx, ny, 0, N)
        i2 = link_coord_to_i(nx + 1, ny, 1, N)
        i3 = link_coord_to_i(nx, ny+1, 0, N)
        i4 = link_coord_to_i(nx, ny, 1, N)

        # fill in number
        mat[i1, j] = 1.
        mat[i2, j] = 1.
        mat[i3, j] = -1.
        mat[i4, j] = -1.
    end

    return mat
end

function get_mat_quad(mat_RtoE)
    return transpose(mat_RtoE) * mat_RtoE
end

function get_mat_quad_N(N)
    return get_mat_quad(get_mat_RtoE(N))
end

function get_t_max(M_quad)
    return 1 / maximum(diag(inv(M_quad)))
end

# %%
m_RtoE = get_mat_RtoE(Ns)
m_EtoR = pinv(m_RtoE)
m_quad = get_mat_quad(m_RtoE)
eta_max = get_t_max(m_quad)
eig_vals, U = eigen(m_quad)
Ut = transpose(U)
@assert isapprox(U * Diagonal(eig_vals) * Ut, m_quad)

# %%
function exact_vacuum_energy(lambdas = eig_vals)
    return 1/2 * sum(sqrt.(lambdas))
end

# %%
eig_vals

# %%
# println("R to E conversion matrix")
# # display(m_RtoE)
# show(stdout, "text/plain", m_RtoE)
# println("\n\nMatrix for the electric Hamiltonian")
# show(stdout, "text/plain", m_quad)
# println("\n\nU matrix is")
# show(stdout, "text/plain", U)
println("\n\nMaximal eta is $(eta_max)")

# %% [markdown]
# Hamiltonian is 
# 
# $$H = \frac{g^2}{2} R_i M_{i j} R_j + \frac{1}{2 g^2 a^2} \sum_p B_p^2 = \frac{\tilde{g}^2}{2 a} R_i M_{i j} R_j + \frac{1}{2 \tilde{g}^2 a} \sum_p B_p^2$$
# 
# Let's denote $M = U \Lambda U^{-1}$, where $\Lambda$ is diagonal matrix with real positive entries. Denote the eigenvalues by $\lambda_i$.
# 
# The action is
# 
# $$- S = \sum_t \left( - \frac{a}{2 \tilde{g}^2 a^0 } \sum_i \frac{1}{\lambda_i}
# (C_i' - C_i)^2 - \frac{a^0}{2\tilde{g}^2 a} \sum_p B_p^2 \right),$$
# where $C_i = (U^\dagger)_{i j} B_j =(U^T)_{i j} B_j$.
# 

# %%
const odd_time = [Float64(i % 2) for i in 1:Nt]
const even_time = 1 .- odd_time;

# %%
# a place to store all the intermediate results
# use pre-allocated arrays to save the allocation cost
struct MCWorkspace
    lat_transformed::Array{Float64, 2}
    lat_temp::Array{Float64, 3}
    fluctuation::Array{Float64, 3}
    S_old::Vector{Float64}
    S_new::Vector{Float64}
    kin_term::Vector{Float64}
    pot_term::Vector{Float64}
    success_flags::Vector{Float64}

    function MCWorkspace(nt::Integer, ns::Integer)
        new(zeros(nt, ns*ns), zeros(nt, ns, ns), zeros(nt, ns, ns),
            zeros(nt), zeros(nt), zeros(nt), zeros(nt), zeros(nt))
    end
end


MCWorkspace() = MCWorkspace(Nt, Ns)

@kwdef struct CorrResult
    px::Int
    py::Int
    vals::Vector{Float64}
end

@kwdef struct MCSingleSetResult
    energies::Vector{Float64}
    B2max_list::Vector{Float64}
    B2_mean::Matrix{Float64}
    R2_mean::Matrix{Float64}
    accept_rate::Float64
    corrs::Vector{CorrResult}
end


struct MCBinningResult
    bin_sizes::Vector{Float64}
    means::Vector{Float64}
    errs::Vector{Float64}
end

struct MCResult
    gs_energy_mean::Float64
    gs_energy_delta::Float64
    gs_binning_result::MCBinningResult
    subtract_ratio::Union{Nothing, Float64}
end

MCResult(gs_energy_mean, gs_energy_delta, gs_binning_result) =
    MCResult(gs_energy_mean, gs_energy_delta, gs_binning_result, nothing)

# %%
# original code

function S_lat_kin_tloc(lat; lambda_list=eig_vals, U_mat=U) # st stands for single time,
    """
    lat: real nd array (Nt, N, N)
        the lattice system
    m_quad: real nd array (N^2, N^2)
        the quadratic form of the energy matrix

    This function computes the action itself, but the part of action that will change.
        This is like the checkered board, except I am just doing it for even and odd time slices instead of 
        even and point configuration points
    """
    lat_view = reshape(lat, (Nt, Ns*Ns))
    lat_C = lat_view * U_mat  # (Nt, Ns*Ns)

    """
    kin_terms = (lat_C .- circshift(lat_C, (-1, 0))) .^ 2 ./ reshape(lambda_list, (1, Ns*Ns))
    return 1 / (2 * g^2 * a_0) * sumr(kin_terms, dims=2) # return shape (Nt)
    """

    kin_terms = similar(lat_C)
    @inbounds for i in 1:Nt
        i_n = mod1(i+1, Nt)
        i_p = mod1(i-1, Nt)
        @simd for j in 1:Ns*Ns
            kin_terms[i, j] = 1 / (g^2 * a_0) * lat_C[i, j] * (lat_C[i, j] - lat_C[i_p, j] - lat_C[i_n, j]) / lambda_list[j]
        end
    end
    
    return sumr(kin_terms, dims=2) # return shape (Nt)
end

function S_lat_pot_tloc(lat)
    return a_0 / 2 / g^2 * sumr(lat .^ 2, dims=(2, 3)) # return shape (Nt)
end


function S_lat_tloc(lat; lambda_list=eig_vals, U_mat=U)
    return S_lat_kin_tloc(lat, lambda_list=lambda_list, U_mat = U_mat) .+ S_lat_pot_tloc(lat)
    # return shape (Nt)
end

# %%
# code that runs a little faster
function S_lat_pot_tloc!(lat, pol_terms_return)
    # pol_terms return have shape (Nt)
    pol_terms_return .= 0.
    @inbounds for k in axes(lat, 3), j in axes(lat, 2)
        pol_terms_return .+= (a_0 / 2 / g^2) .* view(lat, :, j, k).^2
    end
end



function S_lat_kin_tloc!(lat, lat_C, kin_terms_return; lambda_list=eig_vals, U_mat=U) # st stands for single time,
    """
    lat: real nd array (Nt, N, N)
        the lattice system

    kin_terms_return: pre-allocated array of shape (Nt)

    This function computes the action itself, but the part of action that will change.
        This is like the checkered board, except I am just doing it for even and odd time slices instead of 
        even and point configuration points
    """
    lat_view = reshape(lat, (Nt, Ns*Ns))
    # lat_C = lat_view * U_mat  # (Nt, Ns*Ns)
    mul!(lat_C, lat_view, U_mat) # (Nt, Ns*Ns)

    """
    kin_terms = (lat_C .- circshift(lat_C, (-1, 0))) .^ 2 ./ reshape(lambda_list, (1, Ns*Ns))
    return 1 / (2 * g^2 * a_0) * sumr(kin_terms, dims=2) # return shape (Nt)
    """
    
    kin_terms_return .= 0.
    # kin_terms = similar(lat_C)
    @inbounds for j in 1:Ns*Ns
        @simd for i in 1:Nt
            i_n = mod1(i+1, Nt)
            i_p = mod1(i-1, Nt)
            kin_terms_return[i] += 1 / (g^2 * a_0) * lat_C[i, j] * (lat_C[i, j] - lat_C[i_p, j] - lat_C[i_n, j]) / lambda_list[j]
        end
    end
    nothing
    # return sumr(kin_terms, dims=2) # return shape (Nt)
end


function S_lat_tloc!(lat::Array{Float64, 3}, S_lat_draft::Vector{Float64}, wc::MCWorkspace; lambda_list=eig_vals, U_mat=U)
    S_lat_kin_tloc!(lat, wc.lat_transformed, wc.kin_term; lambda_list=lambda_list, U_mat = U_mat)
    S_lat_pot_tloc!(lat, wc.pot_term)
    S_lat_draft .= wc.kin_term .+ wc.pot_term # shape of S_lat_draft is (Nt)
    # return shape (Nt)
    nothing
end

# %%
function update!(lat::Array{Float64, 3}, wc::MCWorkspace, get_S_tloc! = S_lat_tloc!)
    
    wc.fluctuation .= rand.() .* 2 .* eps .- eps
    # fluctuation = rand(Nt, Ns, Ns) .* 2 .* eps .- eps

    success_num = 0

    # now, let me do it alternatively between even layers and odd layers of time slice
    # S_old_odd  = get_S_lat(lat)
    # S_new_odd  = get_S_lat(lat .+ fluctuation .* odd_time)
    # # Now doing odd sites
    get_S_tloc!(lat, wc.S_old, wc)
    wc.lat_temp .= lat .+ wc.fluctuation .* odd_time
    get_S_tloc!(wc.lat_temp, wc.S_new, wc)
    # dS_odd     = S_new_odd .- S_old_odd
    # odd_success = Float64.(exp.(-dS_odd) .> rand(Nt)) .* odd_time
    wc.success_flags .= Float64.(exp.(-wc.S_new .+ wc.S_old) .> rand.()) .* odd_time
    lat .+= wc.fluctuation .* wc.success_flags
    success_num += Int32(sum(wc.success_flags))

    ## Now do even sites
    get_S_tloc!(lat, wc.S_old, wc)
    wc.lat_temp .= lat .+ wc.fluctuation .* even_time
    get_S_tloc!(wc.lat_temp, wc.S_new, wc)
    wc.success_flags .= Float64.(exp.(-wc.S_new .+ wc.S_old) .> rand.()) .* even_time
    lat .+= wc.fluctuation .* wc.success_flags
    success_num += Int32(sum(wc.success_flags))

    return success_num
end


# %%
function Hamiltonian_kin_terms(confs; lambda_list=eig_vals, U_mat=U)
    @assert ndims(confs) == 4 # (N_cf, N_t, N_s, N_s)
    confs_newshape = (size(confs)[1:end-2]..., size(confs)[end-1] * size(confs)[end]) # (N_cf, N_t, N_s * N_s)
    confs_view = reshape(confs, confs_newshape)

    confs_C = similar(confs_view)
    for i in axes(confs_C, 1)
        confs_C[i, :, :] = confs_view[i, :, :] * U_mat
    end
    # confs_C = confs_view * U_mat

    terms_1 = 1 / 2 / a_0
    terms_2 = - ( (confs_C .- circshift(confs_C, (0, 1, 0))) ./ (g*a_0) ).^2 ./ 2 ./ reshape(lambda_list, (1, 1, Ns * Ns)) 
    return terms_1 .+ terms_2 # shape (N_cf, N_t, N_s*N_s)
end

function Hamiltonian_pol_terms(confs)
    return  1 / (2 * g^2) * confs .^ 2 # shape (N_cf, N_t, N_s, N_s)
end

function Hamiltonian_tot(confs; lambda_list=eig_vals, U_mat=U)
    @assert ndims(confs) == 4 # (N_cf, N_t, N_s, N_s)
    H_kin_at_one_time = dropdims(sum(Hamiltonian_kin_terms(confs, lambda_list=lambda_list, U_mat=U_mat), dims=3), dims=3) # (N_cf, N_t)
    H_kin_avgtime = dropdims(mean(H_kin_at_one_time, dims=2), dims=2) # (N_cf)

    H_pol_at_one_time = dropdims(sum(Hamiltonian_pol_terms(confs), dims=(3, 4)), dims=(3, 4)) # (N_cf, N_t)
    H_pol_avgtime = dropdims(mean(H_pol_at_one_time, dims=2), dims=2)
    
    return H_kin_avgtime + H_pol_avgtime # (N_cf)
end


# %%
function estimate_B2(confs, p_x, p_y)
    return mean(confs[:, :, p_x, p_y].^2)
end

function estimate_R2(confs, p_x, p_y; lambda_list=eig_vals, U_mat=U)
    p_i = plaquette_coord_to_i(p_x, p_y, Ns)
    S2_terms = meanr(Hamiltonian_kin_terms(confs; lambda_list, U_mat), dims=(1, 2)) .* 2 ./ g^2 ./ lambda_list # (Ns)
    return sum(U_mat[p_i, :].^2 .* S2_terms)
end

function get_correlator(confs, d, p_x, p_y)
    return mean(confs[:, :, p_x, p_y] .* circshift(confs[:, :, p_x, p_y], (0, d)))
end


function Metropolis_single(S_tloc! = S_lat_tloc! , get_Hamiltonian = Hamiltonian_tot ; Nt::Int = Nt, Ns::Int = Ns,
        n_batches::Int = n_batches, batch_size::Int = batch_size, p::Progress, corrs_watch::Vector{Tuple{Int, Int}})
    lat = zeros(Nt, Ns, Ns)
    paths = zeros(batch_size, Nt, Ns, Ns)

    energies = Array{Float64}(undef, batch_size * n_batches)
    B2max_list = Array{Float64}(undef, batch_size * n_batches)
    B2_mean_ls = Array{Float64}(undef, Ns, Ns, n_batches)
    R2_mean_ls = Array{Float64}(undef, Ns, Ns, n_batches)

    n_corrs = length(corrs_watch)
    corrs_vals_ls = Array{Float64}(undef, Nt, n_batches, n_corrs)
    corrs = Array{CorrResult}(undef, n_corrs)



    tot_updates = 0
    acc_updates = 0

    wc = MCWorkspace(Nt, Ns)

    for j in 1:N_thermal
        sn = update!(lat, wc, S_tloc!)
        tot_updates += Nt
        acc_updates += sn
    end

    for n in 1:n_batches
        for i in 1:batch_size
            for j in 1:N_cor
                sn = update!(lat, wc, S_tloc!)
                tot_updates += Nt
                acc_updates += sn
            end
            paths[i, :, :, :] = lat

            next!(p) # notify the progress meter
        end

        i_s = (n - 1) * batch_size + 1
        i_e = i_s + batch_size - 1
        energies[i_s:i_e] = get_Hamiltonian(paths)
        B2max_list[i_s:i_e] = meanr(dropdims(maximum(paths.^2, dims=(3, 4)), dims=(3, 4)), dims=2)
        B2_mean_ls[:, :, n] = meanr(paths.^2, dims=(1, 2))
        R2_mean_ls[:, :, n] = [estimate_R2(paths, p_x, p_y) for p_x in 1:Ns, p_y in 1:Ns]
        for (nc, corr_indices) in enumerate(corrs_watch)
            px, py = corr_indices
            corrs_vals_ls[:, n, nc] = [get_correlator(paths, t, px, py) for t in 0:Nt-1]
        end
    end
    # if debug
    #     println("Accept rate ", acc_updates/tot_updates)
    # end
    accept_rate = acc_updates/tot_updates
    B2_mean = meanr(B2_mean_ls, dims=3)
    R2_mean = meanr(R2_mean_ls, dims=3)
    for (nc, corr_indices) in enumerate(corrs_watch)
        px, py = corr_indices
        corrs_nc_ls = meanr(corrs_vals_ls[:,:, nc], dims=2)
        corrs[nc] = CorrResult(;px,py,vals=corrs_nc_ls)
    end
    result = MCSingleSetResult(;energies, B2max_list, B2_mean, R2_mean, accept_rate, corrs)
    

    return result
end


function Metropolis_parallel(S_tloc! = S_lat_tloc! , get_Hamiltonian = Hamiltonian_tot ; debug=false, Nt::Int = Nt, Ns::Int = Ns, 
    n_batches::Int = n_batches, batch_size::Int = batch_size, N_sets::Int = N_sets, corrs_watch::Vector{Tuple{Int, Int}})

    N_cf_s = n_batches * batch_size
    N_cf_tot = N_cf_s * N_sets
    p = Progress(N_cf_tot; dt=1.0)

    energies = Array{Float64}(undef, N_cf_tot)
    B2max_list = Array{Float64}(undef, N_cf_tot)
    B2_mean_ls = Array{Float64}(undef, Ns, Ns, N_sets)
    R2_mean_ls = Array{Float64}(undef, Ns, Ns, N_sets)
    accept_rate_ls = Array{Float64}(undef, N_sets)
    n_corrs = length(corrs_watch)
    corrs_vals_ls = Array{Float64}(undef, Nt, N_sets, n_corrs)
    corrs = Array{CorrResult}(undef, n_corrs)

    Threads.@threads for i in 1:N_sets
        i_start = (i-1) * N_cf_s + 1
        i_end = i_start + N_cf_s - 1

        result_i = Metropolis_single(S_tloc! , get_Hamiltonian; Nt, Ns, n_batches, batch_size, p, corrs_watch)

        energies[i_start:i_end] = result_i.energies
        B2max_list[i_start:i_end] = result_i.B2max_list
        B2_mean_ls[:, :, i] = result_i.B2_mean
        R2_mean_ls[:, :, i] = result_i.R2_mean
        accept_rate_ls[i] = result_i.accept_rate
        for (nc, corr_indices) in enumerate(corrs_watch)
            px, py = corr_indices
            corrs_i_nc = result_i.corrs[nc]
            corrs_vals_ls[:, i, nc] = corrs_i_nc.vals
            @assert px == corrs_i_nc.px
            @assert py == corrs_i_nc.py
        end
    end

    B2_mean = meanr(B2_mean_ls, dims=3)
    R2_mean = meanr(R2_mean_ls, dims=3)
    accept_rate = mean(accept_rate_ls)
    for (nc, corr_indices) in enumerate(corrs_watch)
        px, py = corr_indices
        corrs_nc_ls = meanr(corrs_vals_ls[:,:, nc], dims=2)
        corrs[nc] = CorrResult(;px,py,vals=corrs_nc_ls)
    end

    if debug
       println("Accept rate is $(accept_rate)")
    end

    result = MCSingleSetResult(;energies, B2max_list, B2_mean, R2_mean, accept_rate, corrs)
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


# function binning_plateau_analysis(errors; threshold = 0.05, minimum_plateau_len = 2, print_plateau = false)
#     plateau_start, plateau_interval_diff = find_binning_plateau_index(errors; threshold)
#     errors_plateau = view(errors, plateau_start:(plateau_start+plateau_interval_diff))

#     if print_plateau
#         println(errors_plateau)
#     end

#     if plateau_interval_diff < minimum_plateau_len
#         println("Warning: Could not find a plateau with width $(minimum_plateau_len)")
#         println("Max plateau width is $(plateau_interval_diff)")
#     end

#     error_final = maximum(errors_plateau)
#     return error_final
# end


        
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
@time MC_result = Metropolis_parallel(;debug=true, corrs_watch = corr_watch_ls);

# %%
# println(mean(confs))
# println(std(confs))

# %%
# mean(S_lat_tloc(confs[1, :, :, :]))

# %% [markdown]
# The potential part of Hamiltonian can be written as
# 
# $$ \frac{1}{2\tilde{g}^2 a} \sum_p B_p^2$$
# 
# The kinematic part of Hamiltonian is originally $\frac{g^2}{2} R_i M_{i j} R_j$. in path integral formulation, $\langle R_k^2 \rangle$ can be evaluated using
# $$\frac{a}{\tilde{g}^2 a^0 \lambda_k} - \left( \frac{a \Delta C_k}{\tilde{g}^2
# a^0 \lambda_k} \right)^2$$



# %%
gs_unmodified_energies = MC_result.energies
gs_unmodified_energies_mean  = mean(gs_unmodified_energies)
gs_unmodified_energies_std   = std(gs_unmodified_energies)
# gs_unmodified_energies_delta = gs_unmodified_energies_std ./ sqrt(N_cf)
println(gs_unmodified_energies_mean)
println(gs_unmodified_energies_std)
# print(gs_unmodified_energies_delta)

# fig = Figure()
# ax = Axis(fig[1, 1], xlabel="configuration label", ylabel="Energy (1/a)")
# scatter!(ax, collect(1:length(gs_unmodified_energies)), gs_unmodified_energies)
# fig

# %%
print(exact_vacuum_energy())

# %%
bin_sizes, means, errors = binning_analysis(gs_unmodified_energies)
print_binning_results(bin_sizes, means, errors)
gs_unmodified_energies_binning_err = find_err_from_binning(errors; threshold=plateau_threshold)

# %%
println(@sprintf "Ground state energy from Monte Carlo is %.5f±%.5f" gs_unmodified_energies_mean gs_unmodified_energies_binning_err)

# %%
println("Exact ground state energy is $(exact_vacuum_energy())")

B2max_mean = mean(MC_result.B2max_list)
B2_bin_sizes, B2_means, B2_errors = binning_analysis(MC_result.B2max_list)
binning_unmodified_B2max = MCBinningResult(B2_bin_sizes, B2_means, B2_errors)
B2max_unmodified_delta = find_err_from_binning(binning_unmodified_B2max.errs; threshold=plateau_threshold)
# println(mean(B2max))
# println(sqrt(mean(B2max)))

println("-------------------------------")



binning_unmodified = MCBinningResult(bin_sizes, means, errors)
# result_unmodified = MCResult(gs_unmodified_energies_mean, gs_unmodified_energies_binning_err, binning_unmodified)

# %%
gs_unmodified_energies_delta = gs_unmodified_energies_binning_err

# %%
# println(estimate_B2(confs, 1, 1))
# println(estimate_R2(confs, 1, 1))
println("For the unmodified vacuum, we have:")
println("Average <B^2> is $(mean(MC_result.B2_mean))")
println("Average <R^2> is $(mean(MC_result.R2_mean))")



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

for corr_result in MC_result.corrs
    f = plot_correlator(collect(0:Nt-1), corr_result.vals)
    px = corr_result.px
    py = corr_result.py
    save_name = joinpath(fig_path, "$(name)_$(px)_$(py).pdf")
    save(save_name, f)
end

# %%
# nx, ny = 2, 2
# d_ls = collect(0:Nt-1)
# corr_ls = [get_correlator(confs, d, nx, ny) for d in d_ls]


# plot_correlator(d_ls, corr_ls)

# %%
# code that runs a little faster
function S_lat_pot_minusB2_tloc!(lat, pol_terms_return; subtract_ratio=1/2)
    # pol_terms return have shape (Nt)
    pol_terms_return .= 0.
    @inbounds for k in axes(lat, 3), j in axes(lat, 2)
        pol_terms_return .+= (a_0 / 2 / g^2) .* view(lat, :, j, k).^2
    end
    pol_terms_return .-= (a_0 / 2 / g^2) .* maximum(lat .^ 2; dims=(2, 3)) .* subtract_ratio
    # pol_terms_return .-= (a_0 / 2 / g^2) .* view(lat, :, site_k_x, site_k_y).^2 .* subtract_ratio
    nothing
end


function S_lat_minusB2_tloc!(lat::Array{Float64, 3}, S_lat_draft::Vector{Float64}, wc::MCWorkspace; subtract_ratio=1/2, lambda_list=eig_vals, U_mat=U)
    S_lat_kin_tloc!(lat, wc.lat_transformed, wc.kin_term; lambda_list=lambda_list, U_mat = U_mat)
    S_lat_pot_minusB2_tloc!(lat, wc.pot_term; subtract_ratio)
    S_lat_draft .= wc.kin_term .+ wc.pot_term # shape of S_lat_draft is (Nt)
    # return shape (Nt)
    nothing
end

function Hamiltonian_pol_terms_minusB2_one_time(confs; subtract_ratio=1/2)
    B2term_lat = sumr(confs .^ 2, dims=(3,4))
    B2term_lat .-= maximum(confs .^ 2, dims=(3, 4)) .* subtract_ratio
    return 1 / (2 * g^2) * B2term_lat # shape (N_cf, N_t)
end

function Hamiltonian_tot_minusB2(confs; subtract_ratio=1/2, lambda_list=eig_vals, U_mat=U)
    @assert ndims(confs) == 4
    H_kin_at_one_time = dropdims(sum(Hamiltonian_kin_terms(confs, lambda_list=lambda_list, U_mat=U_mat), dims=3), dims=3) # (N_cf, N_t)
    H_kin_avgtime = dropdims(mean(H_kin_at_one_time, dims=2), dims=2) # (N_cf)

    H_pol_at_one_time = Hamiltonian_pol_terms_minusB2_one_time(confs; subtract_ratio)# (N_cf, N_t)
    H_pol_avgtime = dropdims(mean(H_pol_at_one_time, dims=2), dims=2)
    
    return H_kin_avgtime + H_pol_avgtime # (N_cf)
end

# %%
# energy_diff_ref_1_1 = gs_unmodified_energies_mean - mean(Hamiltonian_tot_minusB2(confs, 1, 1))
# energy_diff_ref_1_2 = gs_unmodified_energies_mean - mean(Hamiltonian_tot_minusB2(confs, 1, 2))
# energy_diff_ref_2_1 = gs_unmodified_energies_mean - mean(Hamiltonian_tot_minusB2(confs, 2, 1))
# energy_diff_ref_2_2 = gs_unmodified_energies_mean - mean(Hamiltonian_tot_minusB2(confs, 2, 2))

# println(energy_diff_ref_1_1)
# println(energy_diff_ref_1_2)
# println(energy_diff_ref_2_1)
# println(energy_diff_ref_2_2)

println("-----------\n\n\n\n\n")

#%%
# confs = nothing

# %%
function get_MC_results_minusB2(; to_plot_energy=true, to_plot_correlator=false, subtract_ratio = 1/2, print_binning=false, corrs_watch = corr_watch_ls)
    # p_x, p_y = plaquette_i_to_coord(p_i, Ns)


    S_lat_tmp!(lat::Array{Float64, 3}, S_lat_draft::Vector{Float64}, wc::MCWorkspace) = 
    let subtract_ratio = subtract_ratio
        S_lat_minusB2_tloc!(lat, S_lat_draft, wc; subtract_ratio)
    end

    get_Hamiltonian_tmp(confs::Array{Float64, 4}) = 
    let subtract_ratio = subtract_ratio
        Hamiltonian_tot_minusB2(confs; subtract_ratio)
    end

    println("\n\n---------------------B2--------------------")

    # S_lat_tmp = lat -> S_lat_minusB2_tloc(lat, p_x, p_y, subtract_ratio=subtract_ratio)
    MC_B2_result = Metropolis_parallel(S_lat_tmp!, get_Hamiltonian_tmp; debug=true, corrs_watch)
    # println("Configuration mean $(mean(confs_B2k))")

    gs_minusB2_energies = MC_B2_result.energies
    gs_minusB2_energies_mean  = mean(gs_minusB2_energies)
    gs_minusB2_energies_std   = std(gs_minusB2_energies)
    bin_sizes, means, errors = binning_analysis(gs_minusB2_energies)
    gs_minusB2_energies_delta = find_err_from_binning(errors; threshold=plateau_threshold)
    println("Average energy is $gs_minusB2_energies_mean")
    println("Std of energy is $gs_minusB2_energies_std")
    println("Naive uncertainty of energy: $gs_minusB2_energies_delta")

    
    if to_plot_energy
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="configuration label", ylabel="Energy (1/a)", title="Energy")
        scatter!(ax, collect(1:length(gs_minusB2_energies)), gs_minusB2_energies)
        display(fig)
    end
    

    energy_diff = gs_unmodified_energies_mean - gs_minusB2_energies_mean
    energy_diff_delta = hypot(gs_unmodified_energies_delta, gs_minusB2_energies_delta)
    # energy_diff_3delta = energy_diff + 3 * energy_diff_delta
    B2bound = energy_diff * 2 * g^2 / subtract_ratio
    B2bound_sqrt = B2bound >=0 ? sqrt(B2bound) : NaN
    B2bound_with_delta  = (energy_diff + energy_diff_delta) * 2 * g^2 / subtract_ratio
    B2bound_with_3delta = (energy_diff + 3 * energy_diff_delta) * 2 * g^2 / subtract_ratio
    B2bound_sqrt_with_delta = B2bound_with_delta >=0 ? sqrt(B2bound_with_delta) : NaN
    B2bound_sqrt_with_3delta = B2bound_with_3delta >=0 ? sqrt(B2bound_with_3delta) : NaN
    

    B2max_B2_mean = mean(MC_B2_result.B2max_list)
     
    B2_bin_sizes, B2_means, B2_errors = binning_analysis(MC_B2_result.B2max_list)
    binning_B2_B2max = MCBinningResult(B2_bin_sizes, B2_means, B2_errors)
    B2max_B2_delta = find_err_from_binning(binning_B2_B2max.errs; threshold=plateau_threshold)

    if print_binning
        # bin_sizes, means, errors = binning_analysis(gs_minusB2_energies)
        println("Binning result for the energy")
        print_binning_results(bin_sizes, means, errors)
        println("\n\nBinning result for Bmax^2")
        print_binning_results(B2_bin_sizes, B2_means, B2_errors)
    end

    println("Energy difference is about $energy_diff ± $energy_diff_delta")
    println("This means <B^2> is bounded by $B2bound, whose square root is $B2bound_sqrt")
    println("Or with one delta, <B^2> is bounded by $(B2bound_with_delta), whose square root is $(B2bound_sqrt_with_delta)")
    println("Or with three deltas, <B^2> is bounded by $(B2bound_with_3delta), whose square root is $(B2bound_sqrt_with_3delta)")

    B2_ref = mean(MC_result.B2_mean)
    println("\nAs a reference, for the unmodified Hamiltonian <B^2> is $(B2_ref)")
    println("Also, for the unmodified vacuum, <B_max^2> is $(B2max_mean), the sqrt of which is $(sqrt(B2max_mean))")
    println("Also, for the modified vacuum, <B_max^2> is $(B2max_B2_mean), the sqrt of which is $(sqrt(B2max_B2_mean))")
    println("To be more exact, it's $(B2max_B2_mean)±$(B2max_B2_delta)")
    println("\n\n")


    
    binning_B2_energies = MCBinningResult(bin_sizes, means, errors)

    result_dict = Dict(
        "energy"         => gs_minusB2_energies,
        "energy_mean"    => gs_minusB2_energies_mean,
        "energy_delta"   => gs_minusB2_energies_delta,
        "energy_binning" => binning_B2_energies,
        "B2bound"        => B2bound,
        "subtract_ratio" => subtract_ratio,
        "B2max"          => MC_B2_result.B2max_list,
        "B2max_mean"     => B2max_B2_mean,
        "B2max_delta"    => B2max_B2_delta,
        "B2max_binning"  => binning_B2_B2max
    )
    
    if to_plot_energy
        result_dict["energy fig"] = fig
    end
    
    if to_plot_correlator
        for corr_result in MC_B2_result.corrs
            f = plot_correlator(collect(0:Nt-1), corr_result.vals)
            px = corr_result.px
            py = corr_result.py
            save_name = joinpath(fig_path, "$(name)_B2_$(px)_$(py).pdf")
            save(save_name, f)
        end
    end

    
   

    
    
    return result_dict, MC_B2_result
end

# %%
to_plot_energy = false
to_plot_correlator = true
print_binning = true

# %%


@time result_dic_B2, MC_B2_result = get_MC_results_minusB2(; to_plot_energy, to_plot_correlator, print_binning, subtract_ratio)

#

# %%


HDF5.h5open(data_name, "w") do file
    HDF5.create_group(file, "parameters")
    file["parameters"]["Ns"]  = Ns
    file["parameters"]["Nt"]  = Nt
    file["parameters"]["a_0"] = a_0
    file["parameters"]["g"]   = g

    HDF5.create_group(file, "unmodified")
    file["unmodified"]["energy_mean"] = gs_unmodified_energies_mean
    file["unmodified"]["energy_delta"] = gs_unmodified_energies_binning_err
    file["unmodified"]["B2max"] = mean(MC_result.B2max_list)
    file["unmodified"]["B2max_delta"] = B2max_unmodified_delta
    file["unmodified"]["B2_mean"] = MC_result.B2_mean
    file["unmodified"]["R2_mean"] = MC_result.R2_mean
    HDF5.create_group(file["unmodified"], "binning_energy")
    file["unmodified/binning_energy"]["bin_sizes"] = binning_unmodified.bin_sizes
    file["unmodified/binning_energy"]["means"] = binning_unmodified.means
    file["unmodified/binning_energy"]["errs"] = binning_unmodified.errs


    HDF5.create_group(file, "B2")
    file["B2"]["description"] = "Bounding all B2 once by bounding (B_max)^2"
    file["B2"]["energy_mean"] = result_dic_B2["energy_mean"]
    file["B2"]["energy_delta"] = result_dic_B2["energy_delta"]
    file["B2"]["B2max_mean"] = result_dic_B2["B2max_mean"]
    file["B2"]["B2max_delta"] = result_dic_B2["B2max_delta"]
    file["B2"]["subtract_ratio"] = result_dic_B2["subtract_ratio"]
    HDF5.create_group(file["B2"], "binning_energy")
    HDF5.create_group(file["B2"], "binning_B2max")
    file["B2/binning_energy"]["bin_sizes"] = result_dic_B2["energy_binning"].bin_sizes
    file["B2/binning_energy"]["means"] = result_dic_B2["energy_binning"].means
    file["B2/binning_energy"]["errs"] = result_dic_B2["energy_binning"].errs
    file["B2/binning_B2max"]["bin_sizes"] = result_dic_B2["B2max_binning"].bin_sizes
    file["B2/binning_B2max"]["means"] = result_dic_B2["B2max_binning"].means
    file["B2/binning_B2max"]["errs"] = result_dic_B2["B2max_binning"].errs

    HDF5.create_group(file["unmodified"], "correlations")
    for corr_result in MC_result.corrs
        px = corr_result.px
        py = corr_result.py
        corr_g = HDF5.create_group(file["unmodified/correlations"], "$(px)_$(py)")
        corr_g["px"]   = px
        corr_g["py"]   = py
        corr_g["vals"] = corr_result.vals
    end

    HDF5.create_group(file["B2"], "correlations")
    for corr_result in MC_B2_result.corrs
        px = corr_result.px
        py = corr_result.py
        corr_g = HDF5.create_group(file["B2/correlations"], "$(px)_$(py)")
        corr_g["px"]   = px
        corr_g["py"]   = py
        corr_g["vals"] = corr_result.vals
    end

end
