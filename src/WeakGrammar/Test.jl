import CSV
import DataFrames
using Plots 

include("Inference.jl")
df = DataFrames.DataFrame(CSV.File("files/rps_v1_data.csv"))
gameplayer = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "4df9aef5-2ea7-46b0-ace9-5d19c4532c03"), :]

rps_table = Dict("rock" => 0, "paper" => 1, "scissors" => 2)
outcome_table = Dict("tie" => 0, "win" => 1, "loss" => 2)

df_clean = DataFrames.combine(gameplayer, 
    :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :player_move, 
    :player_outcome => DataFrames.ByRow((x) -> outcome_table[x]) => :player_outcome)

moves_input = Matrix(df_clean)[1:end-1,:]
moves_condition = Matrix(df_clean)[2:end,1]

# println("Inference with SMC")
# traces_smc = unfold_particle_filter(5000, df_clean)
# scores_smc = sort([(get_score(t), i) for (i, t) in enumerate(traces_smc.traces)])
# display(scores_smc[end-9:end])


# println("Inference with SMC and rejuvenation")
# traces_smc_rejuv = unfold_particle_filter_rejuv(10, 3, df_clean)
# scores_smc_rejuv = sort([(Gen.get_score(t), i) for (i,t) in enumerate(traces_smc_rejuv.traces)])
# display(scores_smc_rejuv[end-9:end])

n = 300
M = Matrix{Int64}(undef, n, 2)
M[1,1], M[1,2] = 0,0
for i = 2:n
    if M[i-1,2] == 0
        M[i,1] = (M[i-1,1]+2)%3
    elseif M[i-1,2] == 1
        M[i,1] = M[i-1,1]
    else
        M[i,1] = (M[i-1,1]+1)%3
    end
    # M[i,1] = categorical([1/3,1/3,1/3])-1
    M[i,2] = categorical([1/3,1/3,1/3])-1
end
df_synth = DataFrames.DataFrame(player_move=M[:,1], player_outcome=M[:,2])

# println("Inference with SMC")
# traces_smc = unfold_particle_filter(5000, df_synth)
# scores_smc = sort([(get_score(t), i) for (i, t) in enumerate(traces_smc.traces)])
# display(scores_smc[end-9:end])

println("Inference with SMC and rejuvenation")
traces_smc_rejuv = unfold_particle_filter_rejuv(50, 25, df_synth)
scores_smc_rejuv = sort([(Gen.get_score(t), i) for (i,t) in enumerate(traces_smc_rejuv.traces)])
display(scores_smc_rejuv[end-9:end])

# histogram(map((x) -> size(x[:tree]), traces_smc.traces), bins=(0:1:20))
# softmax([eval_kern(traces_smc_rejuv.traces[10][:tree],i,0,2) for i=0:2])


