using ProgressMeter

include("Proposals.jl")

function initialize_trace(moves_input::Matrix{Int64}, moves_condition::Vector{Int64})::Gen.Trace
    tr, _ = Gen.generate(model, (moves_input,),
        Gen.choicemap([(:moves, i) => moves_condition[i] for i=1:length(moves_condition)]...));
    return tr
end

function get_leaves(choices, base)
    submap = Gen.get_submap(choices, base)
    leaf_addrs = Any[
        (base => address)
        for (address, value) in Gen.get_values_shallow(submap)
        if address != :kernel_type
    ]
    internal_nodes = [
        address
        for (address, value) in Gen.get_submaps_shallow(submap)
    ]
    return cat(leaf_addrs, [
            map((x) -> base => x, get_leaves(submap, node)) for node in internal_nodes]...
        , dims=1)
end


# Inference with MCMC
function run_mcmc(trace::Gen.Trace, iters::Int)::Gen.Trace
    for iter=1:iters
        trace, a = Gen.mh(trace, regen_random_subtree_randomness, (), subtree_involution)
    end
    return trace
end

# Inference with SMC
function unfold_particle_filter(n_particles::Int, df::DataFrames.DataFrame)::Gen.ParticleFilterState
# incorporate involutive MCMC here
    init_obs = Gen.choicemap((:init_move => df.player_move[1]),
        (:init_outcome => df.player_outcome[1]))
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    
    p = Progress(size(df, 1)-1; dt=1.0)
    for t=1:size(df, 1)-1
        maybe_resample!(state, ess_threshold=n_particles/2)
        obs = Gen.choicemap((:chain => t => :move) => df.player_move[t+1],
            (:chain => t => :outcome) => df.player_outcome[t+1])
        particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        next!(p)
    end
    
    return state
end

# Inference with rejuvenation
function unfold_particle_filter_rejuv(n_particles::Int, n_mcmc::Int, df::DataFrames.DataFrame)
    # incorporate involutive MCMC here
    init_obs = Gen.choicemap((:init_move => df.player_move[1]),
        (:init_outcome => df.player_outcome[1]))
    state = initialize_particle_filter(unfold_model_dynamic, (0,), init_obs, n_particles)

    p = Progress(size(df, 1)-1; dt=1.0)
    for t=1:size(df, 1)-1
        
        for i=1:n_particles
            for j=1:n_mcmc
                state.traces[i], accepted = Gen.mh(state.traces[i], regen_random_subtree_randomness, (), subtree_involution)
                
                
                leaf_addrs = get_leaves(get_choices(state.traces[i]), :tree)
                selection = Gen.select(leaf_addrs...)
                if accepted && length(leaf_addrs) > 0
                    state.traces[i], _ = Gen.hmc(state.traces[i], selection)
                end
            end
        end
        
        maybe_resample!(state, ess_threshold=n_particles/2)
        obs = Gen.choicemap((:chain => t => :move) => df.player_move[t+1],
            (:chain => t => :outcome) => df.player_outcome[t+1])
        Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        next!(p)
    end
    
    return state
end


