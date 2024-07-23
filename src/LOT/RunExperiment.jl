import CSV
import JSON
using Statistics
include("Inference.jl")


df = DataFrames.DataFrame(CSV.File("files/rps_v1_data.csv"))
gameplayer = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "33261505-3a41-47b5-b3db-83cb6aeb1a9f"), :]
gameopp = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "4df9aef5-2ea7-46b0-ace9-5d19c4532c03"), :]
gameplayer = DataFrames.combine(gameplayer, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :player_move)
gameopp = DataFrames.combine(gameopp, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :opp_move)
df_brockbank = hcat(gameplayer[1:50,:], gameopp[1:50,:])


j=JSON.parsefile("files/chatgpt.json")
M = Matrix{Int64}(undef, length(j["rounds"]), 2)
for i = 1:length(j["rounds"])
    M[i,1] = rps_table[j["rounds"][i]["player1"]]
    M[i,2] = rps_table[j["rounds"][i]["player2"]]
end
df_chatgpt = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

samples = [10, 20, 50]

data = Dict("chatgpt" => 
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "invtemp" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10)),
    "brockbank" =>
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "invtemp" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10))
)

for (n_samples_i, n_samples) in enumerate(samples)
    for (n_mcmc_i, n_mcmc) in enumerate(samples)
        for (n_particles_i, n_particles) in enumerate(samples)
            for indx=1:10
                println("$n_samples, $n_mcmc, $n_particles, $indx")
                traces_gpt_timed = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_chatgpt[1:n_samples,:])
                scores_gpt = sort([(get_score(t), i) for (i,t) in enumerate(traces_gpt_timed[1].traces)])
                traces_brockbank_timed = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_brockbank[1:n_samples,:])
                scores_brockbank = sort([(get_score(t), i) for (i,t) in enumerate(traces_brockbank_timed[1].traces)])
                data["chatgpt"]["score"][n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_gpt[end][1]
                data["brockbank"]["score"][n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_brockbank[end][1]
                data["chatgpt"]["time"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[2]
                data["brockbank"]["time"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[2]
                data["chatgpt"]["invtemp"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[1].traces[scores_gpt[end][2]][:invtemp]
                data["brockbank"]["invtemp"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[1].traces[scores_brockbank[end][2]][:invtemp]
                data["chatgpt"]["tree"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[1].traces[scores_gpt[end][2]][:tree]
                data["brockbank"]["tree"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[1].traces[scores_brockbank[end][2]][:tree]
            end
        end
    end
end
