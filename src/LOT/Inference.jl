using ProgressMeter

include("Model.jl")

function initialize_trace(moves_input::Matrix{Int64}, moves_condition::Vector{Int64})::Gen.Trace
    tr, _ = Gen.generate(model, (moves_input,),
        Gen.choicemap([(:moves, i) => moves_condition[i] for i=1:length(moves_condition)]...));
    return tr
end

function get_invtemps(choices::ChoiceMap, base::Union{Symbol, Pair})
    submap = Gen.get_submap(choices, base)
    leaf_addrs = Any[
        (base => address)
        for (address, value) in Gen.get_values_shallow(submap)
        if address == :invtemp
    ]
    internal_nodes = [
        address
        for (address, value) in Gen.get_submaps_shallow(submap)
    ]
    return cat(leaf_addrs, [
        map((x) -> base => x, get_invtemps(submap, node)) for node in internal_nodes]...
        , dims=1)
end

function get_invtemps(choices::ChoiceMap)
    submap = choices
    leaf_addrs = Any[
        address
        for (address, value) in Gen.get_values_shallow(submap)
        if address == :invtemp
    ]
    internal_nodes = [
        address
        for (address, value) in Gen.get_submaps_shallow(submap)
    ]
    return cat(leaf_addrs, 
        [get_invtemps(submap, node) for node in internal_nodes]...
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
function unfold_particle_filter(num_particles::Int, df::DataFrames.DataFrame)::Gen.ParticleFilterState
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
function unfold_particle_filter_rejuv(num_particles::Int, n_mcmc::Int, df::DataFrames.DataFrame)
    # incorporate involutive MCMC here
        init_obs = choicemap()
        state = initialize_particle_filter(unfold_model, (0,), init_obs, num_particles)
    
        p = Progress(size(df, 1); dt=1.0)
        for t=1:size(df, 1)
            
            for i=1:num_particles
                for j=1:n_mcmc
                    state.traces[i], = mh(state.traces[i], regen_random_subtree_randomness, (), subtree_involution)

                    ch = get_choices(state.traces[i])
                    invtemps = get_invtemps(ch)
                    state.traces[i], = mh(state.traces[i], gaussian_drift, (invtemps,))
                end
            end
            
            maybe_resample!(state, ess_threshold=num_particles/2)
            obs = Gen.choicemap((:chain => t => :move) => df.player_move[t],
                (:chain => t => :opp_move) => df.opp_move[t])
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
            next!(p)
        end
        return state
    end

