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
    n_particles::Int, 
    df::DataFrames.DataFrame)::Gen.ParticleFilterState
# incorporate involutive MCMC here
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    
    p = Progress(size(df, 1); dt=1.0)
    for t=1:size(df, 1)
        maybe_resample!(state, ess_threshold=n_particles/2)
        obs = Gen.choicemap((:chain => t => :move) => df.player_move[t],
            (:chain => t => :opp_move) => df.opp_move[t])
        particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        next!(p)
    end
    
    return state
end

# one step of particle filter
function particle_filter_one_step!(state, t, player_move, opp_move)
    maybe_resample!(state, ess_threshold=length(state.traces)/2)
    obs = Gen.choicemap((:chain => t => :move) => player_move,
        (:chain => t => :opp_move) => opp_move)
    Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
end

# one step of rejuvenation particle filter
function particle_filter_rejuv!(state, n_mcmc)
    # println(state.traces[1][:tree])
    # println(get_score(state.traces[1]))
    for i=1:length(state.traces)
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
            # println("noise rejuvenation")
            state.traces[i], a = mh(state.traces[i], noise_drift_beta, ())
            # println("gamma rejuvenation")
            for addr in gamma_addrs
                state.traces[i], a = mh(state.traces[i], drift_gamma, (addr,))
            end
        end
    end
    return state
end

# Inference with rejuvenation
function unfold_particle_filter_rejuv(
    n_particles::Int, 
    n_mcmc::Int, 
    df::DataFrames.DataFrame)::Gen.ParticleFilterState
    # incorporate involutive MCMC here
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)

    for t=1:size(df, 1)
        particle_filter_one_step!(state, t, df.player_move[t], df.opp_move[t])
        particle_filter_rejuv!(state, n_mcmc)
        println(best_pred(state)[:tree])
    end
    return state
end