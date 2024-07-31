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

function make_noisy_wsls(noise)
    n = 50
    M = Matrix{Int64}(undef, n, 2)
    M[1,1], M[1,2] = 0,0
    for i = 2:n
        if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
            M[i,1] = (M[i-1,1]+cat0([noise/2, noise/2, 1-noise])) % 3# (M[i-1,1]+2)%3
        elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
            M[i,1] = (M[i-1,1]+cat0([1-noise, noise/2, noise/2])) % 3# M[i-1,1]
        else
            M[i,1] = (M[i-1,1]+cat0([noise/2, 1-noise, noise/2])) % 3# (M[i-1,1]+1)%3
        end
        M[i,2] = cat0([1/3,1/3,1/3])
    end
    df_test = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])
    return df_test
end

noises = [0,0.1,0.2]
samples = [10, 20, 50]
mcmc_steps = [5, 10, 100]
particles = [5, 10, 100]

data = Dict("chatgpt" => 
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "noise" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10)),
    "brockbank" =>
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "noise" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10)),
    "wsls" =>
    Dict("score" => zeros((3,3,3,3,10)),
    "time" => zeros((3,3,3,3,10)),
    "noise" => zeros((3,3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,3,10)),
)

for (noise_i, noise) in enumerate(noises)
    df_wsls = make_noisy_wsls(noise)
    for (n_samples_i, n_samples) in enumerate(samples)
        for (n_mcmc_i, n_mcmc) in enumerate(mcmc_steps)
            for (n_particles_i, n_particles) in enumerate(particles)
                for indx=1:10
                    println("$noise, $n_samples, $n_mcmc, $n_particles, $indx")
                    traces_wsls = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_wsls[1:n_samples,:])
                    scores_wsls = sort([(get_score(t), i) for (i,t) in enumerate(traces_wsls[1].traces)])
                    data["wsls"]["score"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_wsls[end][1]
                    data["wsls"]["time"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_wsls[2]
                    data["wsls"]["noise"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_wsls[1].traces[scores_wsls[end][2]][:noise]
                    data["wsls"]["tree"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_wsls[1].traces[scores_wsls[end][2]][:tree]
                end
            end
        end
    end
end

for (n_samples_i, n_samples) in enumerate(samples)
    for (n_mcmc_i, n_mcmc) in enumerate(mcmc_steps)
        for (n_particles_i, n_particles) in enumerate(particles)
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
                data["chatgpt"]["noise"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[1].traces[scores_gpt[end][2]][:noise]
                data["brockbank"]["noise"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[1].traces[scores_brockbank[end][2]][:noise]
                data["chatgpt"]["tree"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[1].traces[scores_gpt[end][2]][:tree]
                data["brockbank"]["tree"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[1].traces[scores_brockbank[end][2]][:tree]
            end
        end
    end
end