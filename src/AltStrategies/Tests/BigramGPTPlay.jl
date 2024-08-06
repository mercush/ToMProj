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

function BigramPlay(n)
    table = fill(0.1,(3,3))
    score = zeros(2)
    model = "gpt-4-turbo"

    dialog = []
    prompt = "You and I are going to play a game of rock, paper, scissors. Make a move"
    push!(dialog, Dict("role" => "user", "content" => prompt))

    s = 1
    p = Progress(n, 2)
    prev = 0
    for t=1:n


        best_response = argmax(table[prev+1, :])%3


        gpt_response = create_chat(
            ENV["OPENAI_API_KEY"],
            model,
            dialog
        )
        gpt_message = gpt_response.response[:choices][begin][:message][:content]
        push!(dialog, Dict("role" => "assistant", "content" => gpt_message))
        gpt_move = parse_message(gpt_message)
        

        likelihood = table[prev+1, gpt_move+1]/sum(table[prev+1,:])
        if likelihood < 0.05
            table = fill(0.1,(3,3))
        end

        table[prev+1, gpt_move+1] += 1

        outcome = (3+gpt_move-best_response)%3
        if outcome != 0
            score[outcome] += 1
        end
        prompt = "I make move $(inv_transform[best_response]). Make your next move."
        push!(dialog, Dict("role" => "user", "content" => prompt))

        prev = gpt_move
        next!(p)
    end
    return score
end

gpt_score = BigramPlay(50)