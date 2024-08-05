include("Inference.jl")

function parse_message(message)
    lower_message = lowercase(message)
    for a in ["rock", "paper", "scissors"]
        if occursin(a, lower_message)
            return a
        end
    end
end

function ChatGPTTest(n, n_particles, n_mcmc)
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    score = 0
    model = "gpt-3.5-turbo"

    prompts = []
    prompt = "You and I are going to play a game of rock, paper, scissors. Make a move"
    push!(prompts, Dict("role" => "user", "content" => prompt))

    for t=1:n

        gpt_response = create_chat(
            secret_key,
            model,
            [Dict("role" => "user", "content"=> prompt)]
        )
        gpt_move = parse_message(gpt_response)

        player_move = transform[m]
        next_move = next_move_pred(state)
        println("I thought your next move distribution was ", next_move)
        opp_move = argmax(next_move)%3
        score += (4+player_move-opp_move)%3-1
        particle_filter_one_step!(state, t, player_move, opp_move)
        particle_filter_rejuv!(state, n_mcmc)

        if next_move[player_move+1] < 0.1
            init_obs = choicemap((:chain => 1 => :move, player_move), (:chain => 1 => :opp_move, opp_move))
            state = initialize_particle_filter(unfold_model, (1,), init_obs, n_particles)
        end
    end
    return score
end

gpt_score = ChatGPTTest(50, 50, 10)