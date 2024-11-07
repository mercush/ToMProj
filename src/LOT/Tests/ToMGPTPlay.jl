using OpenAI

include("../Inference.jl")


transform = Dict("rock" => 0, "paper" => 1, "scissors" => 2)
inv_transform = Dict(0 => "rock", 1 => "paper", 2 => "scissors")
function parse_message(message)
    lower_message = lowercase(message)
    for a in ["rock", "paper", "scissors"]
        if occursin(a, lower_message)
            return transform[a]
        end
    end
    println(message)
    error("ChatGPT made invalid move")
end

function ChatGPTTest(n, n_particles, n_mcmc)
    init_obs = choicemap()
    state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
    score = zeros(2)
    model = "gpt-4-turbo"

    dialog = []
    prompt = "You and I are going to play a game of rock, paper, scissors. Make a move. Answer by simply saying \'rock, paper, or scisors\'. "
    push!(dialog, Dict("role" => "user", "content" => prompt))

    s = 1
    p = Progress(n, 2)
    for t=1:n

        gpt_response = create_chat(
            ENV["OPENAI_API_KEY"],
            model,
            dialog
        )
        gpt_message = gpt_response.response[:choices][begin][:message][:content]
        push!(dialog, Dict("role" => "assistant", "content" => gpt_message))
        gpt_move = parse_message(gpt_message)
        next_move = next_move_pred(state)
        opp_move = argmax(next_move)%3
        outcome = (3+gpt_move-opp_move)%3
        if outcome != 0
            score[outcome] += 1
        end
        prompt = "I make move $(inv_transform[opp_move]). Make your next move."
        push!(dialog, Dict("role" => "user", "content" => prompt))

        if next_move[gpt_move+1] < 0.1
            state = initialize_particle_filter(unfold_model, (0,), init_obs, n_particles)
            s = 1
        end

        particle_filter_one_step!(state, s, gpt_move, opp_move)
        particle_filter_rejuv!(state, n_mcmc)

        s += 1
        next!(p)
    end
    return score
end

gpt_score = ChatGPTTest(50, 50, 10)