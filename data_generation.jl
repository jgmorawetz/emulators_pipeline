using Distributed
using NPZ
using SlurmClusterManager
using EmulatorsTrainer
using JSON3
using Random
using LinearAlgebra
using PyCall


ENV["SLURM_NTASKS"] = ENV["JULIA_TOTAL_TASKS"]
mgr = SlurmManager(;launch_timeout = 600.0, srun_post_exit_sleep = 2.0)
addprocs(mgr)


@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, LinearAlgebra, PyCall
end


@everywhere begin

    # Specify the emulator input parameters and their lower/upper bounds in the desired order (needs to be adjusted by model type)
    pars = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"]
    lb = [0.2, 2.3, 0.8, 50, 0.02, 0.08, 0, -3, -3]
    ub = [1.6, 3.7, 1.1, 90, 0.025, 0.18, 0.5, 1, 2]

    # Specify the desired number of data samples (note: if w0wa extension is considered, some samples will be discarded for w0+wa>0)
    n = 10000

    # Specify the random seed for reproducibility purposes (or set to nothing otherwise)
    seed = nothing
    if seed != nothing
        Random.seed!(seed)
    end

    # Uses latin hypercube sampling, and removes samples with w0+wa>0 since unphysical
    # Must change the indices to match positions of w0,wa in the input vector!
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    w0_ind, wa_ind = 8, 9
    s_cond = [s[w0_ind, i] + s[wa_ind, i] for i in 1:n] 
    s = s[:, s_cond.<0.0]

    # Specify the directory path to store training samples (recommended to change depending on which model/code being used)
    root_dir = "/pscratch/sd/j/jgmorawe/ace_class_mnuw0wacdm_" * string(n)
    
    # Import necessary python modules from Julia
    classy = pyimport("classy")
    np = pyimport("numpy")


    # Function which takes in the training sample parameters and saves statistics to file
    function classy_script(CosmoDict, root_path)
        try
            # Creates subfolders to store each training sample in (uses random string for this)
            rand_str = root_path * "/" * randstring(10)
            # Dictionary of the input features to Class (may need to change depending on model)
            z = CosmoDict["z"]
            cosmo_params = Dict(
                "output" => "mPk",
                "P_k_max_h/Mpc" => 20.0,
                "z_pk" => "0.0,3.",
                "ln10^{10}A_s" => CosmoDict["ln10As"],
                "n_s" => CosmoDict["ns"],
                "h" => CosmoDict["H0"] / 100,
                "omega_b" => CosmoDict["ombh2"],
                "omega_cdm" => CosmoDict["omch2"],
                "m_ncdm" => CosmoDict["Mnu"],
                "w0_fld" => CosmoDict["w0"],
                "wa_fld" => CosmoDict["wa"],
                "tau_reio" => 0.0568,
                "N_ur" => 2.0308,
                "N_ncdm" => 1,
                "use_ppf" => "yes",
                "fluid_equation_of_state" => "CLP",
                "cs2_fld" => 1,
                "Omega_Lambda" => 0,
                "Omega_scf" => 0)
            # Initializes the Class object and then computes statistics
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()
            # Computes sigma8 at z=0
            sigma8 = cosmo.sigma8
            CosmoDict["sigma8"] = sigma8
            # Computes sigma8 at z
            sigma8_z = cosmo.sigma(8.0 / (CosmoDict["H0"] / 100), z)
            # Sound horizon at drag epoch
            r_drag = cosmo.rs_drag
            # Background quantities at redshift z
            H_z = cosmo.Hubble(z) * 299792.458
            r_z = cosmo.comoving_distance(z)
            # Growth factor D(z) and growth rate f(z)
            D_z = cosmo.scale_independent_growth_factor(z)
            f_z = cosmo.scale_independent_growth_factor_f(z)
            cosmo.struct_cleanup()
            cosmo.empty()
            # Results outputted for each training sample
            result_sigma8_basis = [CosmoDict["ln10As"], sigma8_z, r_drag, H_z, r_z, D_z, f_z]
            result_ln10As_basis = [sigma8, sigma8_z, r_drag, H_z, r_z, D_z, f_z]

            if any(isnan, result_sigma8_basis) || any(isnan, result_ln10As_basis)
                @error "There are nan values!"
            else
                mkdir(rand_str)
                npzwrite(rand_str * "/result_sigma8_basis.npy", result_sigma8_basis)
                npzwrite(rand_str * "/result_ln10As_basis.npy", result_ln10As_basis)
                open(rand_str * "/effort_dict.json", "w") do io
                    JSON3.write(io, CosmoDict)
                end
            end
        catch e
            println("Something went wrong during calculation!")
            println(CosmoDict)
        end
    end
end

EmulatorsTrainer.compute_dataset(s, pars, root_dir, classy_script, :distributed)