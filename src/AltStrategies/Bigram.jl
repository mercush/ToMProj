function play()
    table = fill(0.1,(3,3))
    transform = Dict("r" => 0, "p" => 1, "s" => 2)
    inv_transform = Dict(0 => "r", 1 => "p", 2 => "s")
    score = 0
    prev = 0
    while true
        best_response = argmax(table[prev+1, :])%3
        println("Make a move")
        m = "T"
        while !(m in ["r", "p", "s"])
            m = readline()
        end
        m = transform[m]
        println("I make move ", inv_transform[best_response])

        likelihood = table[prev+1, m+1]/sum(table[prev+1,:])
        if likelihood < 0.05
            table = fill(0.1,(3,3))
        end

        table[prev+1, m+1] += 1
        score += (4+m-best_response)%3-1
        println("Score is $score")
        println("Likelihood is $likelihood")
        prev = m
    end
end

play()