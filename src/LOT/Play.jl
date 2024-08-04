include("Inference.jl")

function play(n_particles, n_mcmc)
    transform = Dict("r" => 0, "p" => 1, "s" => 2)
    inv_transform = Dict(0 => "r", 1 => "p", 2 => "s")
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    t = 1
    score = 0
    while true
        tr = best_pred(state)
        println("Make a move")
        m = "T"
        while !(m in ["r", "p", "s"])
            m = readline()
        end
        player_move = transform[m]
        next_move = next_move_pred(state)
        println("I thought your next move distribution was ", next_move)
        opp_move = argmax(next_move)%3
        score += (4+player_move-opp_move)%3-1
        println("I make move $(inv_transform[opp_move])")
        println("Score is $score")
        particle_filter_one_step!(state, t, player_move, opp_move)
        particle_filter_rejuv!(state, n_mcmc)
        println("My best prediction for your strategy is ", tr[:tree])
        println("with noise ", tr[:noise])
        println("log marginal probability is ", get_score(tr))

        if next_move[player_move+1] < 0.1
            println("Strategy change detected")
            init_obs = choicemap((:chain => 1 => :move, player_move), (:chain => 1 => :opp_move, opp_move))
            state = initialize_particle_filter(unfold_model, (1,), init_obs, n_particles)
        end
        t += 1
    end
end

play(50, 50)