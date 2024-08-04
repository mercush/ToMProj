import CSV
import JSON
using Statistics
include("Inference.jl")


# brockbank
df = DataFrames.DataFrame(CSV.File("files/rps_v1_data.csv"))
gameplayer = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "33261505-3a41-47b5-b3db-83cb6aeb1a9f"), :]
gameopp = df[(df.game_id .== "ef8a8513-5542-4c94-bfc4-fecbfb74d540") .&& (df.player_id .== "4df9aef5-2ea7-46b0-ace9-5d19c4532c03"), :]
gameplayer = DataFrames.combine(gameplayer, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :player_move)
gameopp = DataFrames.combine(gameopp, :player_move => DataFrames.ByRow((x) -> rps_table[x]) => :opp_move)
df_brockbank = hcat(gameplayer[1:50,:], gameopp[1:50,:])

# chatgpt
j=JSON.parsefile("files/chatgpt.json")
M = Matrix{Int64}(undef, length(j["rounds"]), 2)
for i = 1:length(j["rounds"])
    M[i,1] = rps_table[j["rounds"][i]["player1"]]
    M[i,2] = rps_table[j["rounds"][i]["player2"]]
end
df_chatgpt = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# wsls
n = 50
M = Matrix{Int64}(undef, n, 2)
M[1,1], M[1,2] = 0,0
for i = 2:n
    if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
        M[i,1] = (M[i-1,1]+2) % 3# (M[i-1,1]+2)%3
    elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
        M[i,1] = (M[i-1,1]) % 3# M[i-1,1]
    else
        M[i,1] = (M[i-1,1]+1) % 3# (M[i-1,1]+1)%3
    end
    M[i,2] = cat0([1/3,1/3,1/3])
end
df_wsls = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# random
n = 50
M = Matrix{Int64}(undef, n, 2)
M[1,1], M[1,2] = 0,0
for i = 2:n
    if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
        M[i,1] = (M[i-1,1]+2) % 3# (M[i-1,1]+2)%3
    elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
        M[i,1] = (M[i-1,1]) % 3# M[i-1,1]
    else
        M[i,1] = cat0([1/6, 2/6, 3/6])# (M[i-1,1]+1)%3
    end
    M[i,2] = cat0([1/3,1/3,1/3])
end
random = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# partly random 1
n = 50
M = Matrix{Int64}(undef, n, 2)
M[1,1], M[1,2] = 0,0
for i = 2:n
    if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
        M[i,1] = (M[i-1,1]+2) % 3# (M[i-1,1]+2)%3
    elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
        M[i,1] = (M[i-1,1]) % 3# M[i-1,1]
    else
        M[i,1] = cat0([1/6, 2/6, 3/6])# (M[i-1,1]+1)%3
    end
    M[i,2] = cat0([1/3,1/3,1/3])
end
pr1 = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

# partly random 2
n = 50
M = Matrix{Int64}(undef, n, 2)
M[1,1], M[1,2] = 0,0
for i = 2:n
    if bernoulli(0.5)
        if (3+M[i-1,1]-M[i-1,2]) % 3 == 0
            M[i,1] = (M[i-1,1]+2) % 3# (M[i-1,1]+2)%3
        elseif (3+M[i-1,1]-M[i-1,2]) % 3 == 1
            M[i,1] = (M[i-1,1]) % 3# M[i-1,1]
        else
            M[i,1] = cat0([1/6, 2/6, 3/6])# (M[i-1,1]+1)%3
        end
        M[i,2] = cat0([1/3,1/3,1/3])
    else
        M[i,1] = cat0([1/3,1/3,1/3])
        M[i,2] = cat0([1/3,1/3,1/3])
    end
end
pr2 = DataFrames.DataFrame(player_move=M[:,1], opp_move=M[:,2])

samples = [5, 20, 50]
mcmc_steps = [5, 10, 50]
particles = [5, 10, 50]

data = Dict("chatgpt" => 
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10)),
    "brockbank" =>
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "noise" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10)),
    "wsls" =>
    Dict("score" => zeros((3,3,3,10)),
    "time" => zeros((3,3,3,10)),
    "noise" => zeros((3,3,3,10)),
    "tree" => Array{Node}(undef, 3,3,3,10)),
)

for (n_samples_i, n_samples) in enumerate(samples)
    for (n_mcmc_i, n_mcmc) in enumerate(mcmc_steps)
        for (n_particles_i, n_particles) in enumerate(particles)
            for indx=1:10
                println("$n_samples, $n_mcmc, $n_particles, $indx")
                traces_gpt_timed = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_chatgpt[1:n_samples,:])
                scores_gpt = sort([(get_score(t), i) for (i,t) in enumerate(traces_gpt_timed[1].traces)])
                traces_brockbank_timed = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_brockbank[1:n_samples,:])
                scores_brockbank = sort([(get_score(t), i) for (i,t) in enumerate(traces_brockbank_timed[1].traces)])
                traces_wsls = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_wsls)
                scores_wsls = sort([(get_score(t), i) for (i,t) in enumerate(traces_wsls[1].traces)])
                traces_random = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_random)
                scores_random = sort([(get_score(t), i) for (i,t) in enumerate(traces_random[1].traces)])
                traces_pr1 = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_pr1)
                scores_pr1 = sort([(get_score(t), i) for (i,t) in enumerate(traces_pr1[1].traces)])
                traces_pr2 = @timed unfold_particle_filter_rejuv(n_particles, n_mcmc, df_pr2)
                scores_pr2 = sort([(get_score(t), i) for (i,t) in enumerate(traces_pr2[1].traces)])

                data["chatgpt"]["score"][n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_gpt[end][1]
                data["brockbank"]["score"][n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_brockbank[end][1]
                data["wsls"]["score"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_wsls[end][1]
                data["random"]["score"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_random[end][1]
                data["pr1"]["score"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_pr1[end][1]
                data["pr2"]["score"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = scores_pr2[end][1]



                data["chatgpt"]["time"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[2]
                data["brockbank"]["time"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[2]
                data["wsls"]["time"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_wsls[2]
                data["random"]["time"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_random[2]
                data["pr1"]["time"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_pr1[2]
                data["pr2"]["time"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_pr2[2]


                data["chatgpt"]["noise"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[1].traces[scores_gpt[end][2]][:noise]
                data["brockbank"]["noise"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[1].traces[scores_brockbank[end][2]][:noise]
                data["wsls"]["noise"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_wsls[1].traces[scores_wsls[end][2]][:noise]
                data["random"]["noise"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_random[1].traces[scores_random[end][2]][:noise]
                data["pr1"]["noise"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_pr1[1].traces[scores_pr1[end][2]][:noise]
                data["pr2"]["noise"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_pr2[1].traces[scores_pr2[end][2]][:noise]


                data["chatgpt"]["tree"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_gpt_timed[1].traces[scores_gpt[end][2]][:tree]
                data["brockbank"]["tree"][n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_brockbank_timed[1].traces[scores_brockbank[end][2]][:tree]
                data["wsls"]["tree"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_wsls[1].traces[scores_wsls[end][2]][:tree]
                data["random"]["tree"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_random[1].traces[scores_random[end][2]][:tree]                
                data["pr1"]["tree"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_pr1[1].traces[scores_pr1[end][2]][:tree]                
                data["pr2"]["tree"][noise_i, n_samples_i, n_mcmc_i, n_particles_i, indx] = traces_pr2[1].traces[scores_pr2[end][2]][:tree]                
            end
        end
    end
end