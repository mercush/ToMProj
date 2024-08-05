import CSV
include("../Inference.jl")

rps_table = Dict("rock" => 0, "paper" => 1, "scissors" => 2)
df = DataFrames.DataFrame(CSV.File("files/rps_v1_data.csv"))
gameopp = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "33261505-3a41-47b5-b3db-83cb6aeb1a9f"), :]
gameplayer = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "4df9aef5-2ea7-46b0-ace9-5d19c4532c03"), :]
gameopp = DataFrames.combine(gameopp, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :opp_move)
gameplayer = DataFrames.combine(gameplayer, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :player_move)
df_test = hcat(gameplayer, gameopp)


function fitToM(df, n_particles, n_mcmc)
    """
    Assumes player is playing according to the strategy that the model infers
    """
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    log_likelihood = 0
    p = Progress(size(df, 1), 1)
    for t=1:size(df, 1)
        next_move = next_move_pred(state)
        player_move = df.player_move[t]
        log_likelihood += log(next_move[player_move+1])
        opp_move = df.opp_move[t]
        particle_filter_one_step!(state, t, player_move, opp_move)
        particle_filter_rejuv!(state, n_mcmc)
        
        if next_move[player_move+1] < 0.1
            init_obs = choicemap((:chain => 1 => :move, player_move), (:chain => 1 => :opp_move, opp_move))
            state = initialize_particle_filter(unfold_model, (1,), init_obs, n_particles)
        end
        next!(p)
    end
    return log_likelihood
end

function fitToMAdversary(df, n_particles, n_mcmc)
    """
    Assumes the opponent is trying to infer the next move of the player and makes move to dominate player
    """
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    log_likelihood = 0
    p = Progress(size(df, 1), 1)
    s = 1
    for t=1:size(df, 1)
        next_move = next_move_pred(state)
        player_move = df.player_move[t]
        opp_move = df.opp_move[t]
        log_likelihood += log(next_move[((2+opp_move)%3)+1])
        
        if next_move[player_move+1] < 0.1
            state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
            s = 1
        end

        particle_filter_one_step!(state, s, player_move, opp_move)
        particle_filter_rejuv!(state, n_mcmc)
        s += 1
        next!(p)
    end
    return log_likelihood
end

function best_move_likelihood(df)
    dist = [sum(df.player_move .== i) for i=0:2]
    return sum(dist .* log.(dist/size(df, 1)))
end

function best_transition_likelihood(df)
    dist = zeros(3)
    for t=2:size(df, 1)
        dist[((3 .+ df.player_move[t] - df.player_move[t-1])%3)+1] += 1
    end
    return sum(dist .* log.(dist/size(df, 1)))+log(1/3)
end

ToM = fitToM(df_test, 50, 10)
ToMadv = fitToMAdversary(df_test, 50, 10)
best_move = best_move_likelihood(df_test)
best_transition = best_transition_likelihood(df_test)
println(ToM)
println(ToMadv)
println(best_move)
println(best_transition) 


# Next test is to run this algorithm against chatgpt
