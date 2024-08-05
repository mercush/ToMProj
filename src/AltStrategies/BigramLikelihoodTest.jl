import CSV

rps_table = Dict("rock" => 0, "paper" => 1, "scissors" => 2)
df = DataFrames.DataFrame(CSV.File("files/rps_v1_data.csv"))
gameopp = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "4df9aef5-2ea7-46b0-ace9-5d19c4532c03"), :]
gameplayer = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "33261505-3a41-47b5-b3db-83cb6aeb1a9f"), :]
gameopp = DataFrames.combine(gameopp, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :opp_move)
gameplayer = DataFrames.combine(gameplayer, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :player_move)
df_test = hcat(gameplayer, gameopp)


function restart()
    fill(0.1,(3,3))
end
function fitBigram(df)
    table = fill(0.1,(3,3))
    prev = 0
    log_likelihood = 0
    for t=1:size(df, 1)
        m = df.player_move[t]
        likelihood = table[prev+1, m+1]/sum(table[prev+1,:])
        log_likelihood += log(likelihood)
        if likelihood < 0.05
            table = restart()
        end

        table[prev+1, m+1] += 1
        prev = m
    end
    return log_likelihood
end

function fitBigramAdv(df)
    table = fill(0.1,(3,3))
    prev = 0
    log_likelihood = 0
    for t=1:size(df, 1)
        m = df.player_move[t]
        opp_move = df.opp_move[t]
        player_likelihood = table[prev+1, ((2+opp_move)%3)+1]/sum(table[prev+1,:])
        log_likelihood += log(table[prev+1, ((2+opp_move)%3)+1]/sum(table[prev+1,:]))
        if player_likelihood < 0.05
            table = restart()
        end

        table[prev+1, m+1] += 1
        prev = m
    end
    return log_likelihood
end

println(fitBigram(df_test))
println(fitBigramAdv(df_test))

# -431.81308231617317
# -352.57451253857204