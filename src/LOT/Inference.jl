using ProgressMeter

include("Model.jl")

function initialize_trace(
    moves_input::Matrix{Int64}, 
    moves_condition::Vector{Int64})::Gen.Trace
    tr, _ = Gen.generate(model, (moves_input,),
        Gen.choicemap([(:moves, i) => moves_condition[i] for i=1:length(moves_condition)]...));
    return tr
end

# Inference with MCMC
function run_mcmc(trace::Gen.Trace, iters::Int)::Gen.Trace
    for iter=1:iters
        trace, a = Gen.mh(trace, regen_random_subtree_randomness, (), subtree_involution)
    end
    return trace
end

# Inference with SMC
function unfold_particle_filter(
    num_particles::Int, 
    df::DataFrames.DataFrame)::Gen.ParticleFilterState
# incorporate involutive MCMC here
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, num_particles)
    
    p = Progress(size(df, 1); dt=1.0)
    for t=1:size(df, 1)
        maybe_resample!(state, ess_threshold=num_particles/2)
        obs = Gen.choicemap((:chain => t => :move) => df.player_move[t],
            (:chain => t => :opp_move) => df.opp_move[t])
        particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        next!(p)
    end
    
    return state
end

# Inference with rejuvenation
function unfold_particle_filter_rejuv(
    num_particles::Int, 
    n_mcmc::Int, 
    df::DataFrames.DataFrame)::Gen.ParticleFilterState
    # incorporate involutive MCMC here
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, num_particles)

    # p = Progress(size(df, 1); dt=1.0)
    for t=1:size(df, 1)
        maybe_resample!(state, ess_threshold=num_particles/2)
        obs = Gen.choicemap((:chain => t => :move) => df.player_move[t],
            (:chain => t => :opp_move) => df.opp_move[t])
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        for i=1:num_particles
            ch = get_choices(state.traces[i])
            gamma_addrs = get_gamma(ch, :tree)
            for j=1:n_mcmc
                # println("tree rejuvenation")
                state.traces[i], a = mh(
                    state.traces[i], 
                    regen_random_subtree_randomness, 
                    (), 
                    subtree_involution)
                if a
                    ch = get_choices(state.traces[i])
                    gamma_addrs = get_gamma(ch, :tree)
                end
                # println("move rejuvenation")
                for s=1:t
                    state.traces[i], a = mh(state.traces[i], rejuv_move, (s,))
                end
                # println("noise rejuvenation")
                state.traces[i], a = mh(state.traces[i], noise_drift_beta, ())
                # println("gamma rejuvenation")
                for addr in gamma_addrs
                    state.traces[i], a = mh(state.traces[i], drift_gamma, (addr,))
                end
            end
        end
        # next!(p)
    end
    return state
end

