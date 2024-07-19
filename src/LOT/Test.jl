import CSV

include("Inference.jl")

# df = DataFrames.DataFrame(CSV.File("files/rps_v1_data.csv"))
# gameplayer = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "4df9aef5-2ea7-46b0-ace9-5d19c4532c03"), :]
# gameopp = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "33261505-3a41-47b5-b3db-83cb6aeb1a9f"), :]
# rps_table = Dict("rock" => 0, "paper" => 1, "scissors" => 2)
# gameplayer = DataFrames.combine(gameplayer, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :player_move)
# gameopp = DataFrames.combine(gameopp, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :opp_move)
# df_clean = hcat(gameplayer, gameopp)

# Strategy 1
# n = 300
# M = Matrix{Int64}(undef, n, 2)
# M[1,1], M[1,2] = 0,0
# for i = 2:n
#     if M[i-1,2] == 0
#         M[i,1] = (M[i-1,1]+2)%3
#     elseif M[i-1,2] == 1
#         M[i,1] = M[i-1,1]
#     else
#         M[i,1] = (M[i-1,1]+1)%3
#     end
#     M[i,2] = categorical([1/3,1/3,1/3])-1
# end
# df_test = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# WSLS 
# n = 300
# M = Matrix{Int64}(undef, n, 2)
# M[1,1], M[1,2] = 0,0
# for i = 2:n
#     if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
#         M[i,1] = (M[i-1,1]+2)%3
#     elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
#         M[i,1] = M[i-1,1]
#     else
#         M[i,1] = (M[i-1,1]+1)%3
#     end
#     M[i,2] = categorical([1/3,1/3,1/3])-1
# end
# df_test = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# Partly random
n = 300
M = Matrix{Int64}(undef, n, 2)
M[1,1], M[1,2] = 0,0
for i = 2:n
    if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
        M[i,1] = categorical([1/3,1/3,1/3])-1
    elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
        M[i,1] = M[i-1,1]
    else
        M[i,1] = (M[i-1,1]+1)%3
    end
    M[i,2] = categorical([1/3,1/3,1/3])-1
end
df_test = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# RANDOM
# n = 300
# M = Matrix{Int64}(undef, n, 2)
# M[1,1], M[1,2] = 0,0
# for i = 2:n
#     M[i,1] = categorical([1/3,1/3,1/3])-1
#     M[i,2] = categorical([1/3,1/3,1/3])-1
# end
# df_test = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# 

# println("Inference with SMC")
# traces_smc = unfold_particle_filter(5000, df_test)
# scores_smc = sort([(get_score(t), i) for (i, t) in enumerate(traces_smc.traces)])
# display(scores_smc[end-9:end])


println("Inference with SMC and rejuvenation")
traces_smc_rejuv = unfold_particle_filter_rejuv(100, 10, df_test)
scores_smc_rejuv = sort([(Gen.get_score(t), i) for (i,t) in enumerate(traces_smc_rejuv.traces)])
display(scores_smc_rejuv[end-9:end])

t = traces_smc_rejuv.traces[scores_smc_rejuv[end][2]]
table=Matrix{Int64}(undef, 3,3)
for i=0:2
    for j=0:2
        table[i+1,j+1] = eval_kern(traces_smc_rejuv.traces[scores_smc_rejuv[end][2]][:tree], State([i],[j]))
    end
end
println(t[:certainty])
println(t[:tree])
display(table)

# opt = choicemap(
#     (:node_type, MakeMove:move_type => :node_type, Random),
#     (:move_type => :expr1 => :node_type, CustomZ3),
#     (:move_type => :expr1 => :param_z3, 0),
#     (:move_type => :expr2 => :node_type, CustomZ3),
#     (:move_type => :expr2 => :param_z3, 0),
#     (:move_type => :expr3 => :node_type, CustomZ3),
#     (:move_type => :expr3 => :param_z3, 0),
# )