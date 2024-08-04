function play()
    transform = Dict("R" => 0, "P" => 1, "S" => 2)
    inv_transform = Dict(0 => "R", 1 => "P", 2 => "S")
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
        opp_move = inv_transform[cat0([1/3,1/3,1/3])]
        score += (4+player_move-opp_move)%3-1
        println("I make move $(inv_transform[opp_move])")
        println("Score is $score")
        particle_filter_one_step!(state, t, player_move, opp_move)
        particle_filter_rejuv!(state, n_mcmc)
        t += 1
        println("I think your strategy is ", tr[:tree])
    end
end

play()