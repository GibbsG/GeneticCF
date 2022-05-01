using Pkg; Pkg.activate(".")
using GeCo;
using JLD

include("fico-setup.jl")


res = Vector{Array{Array{Tuple{String,Int64},1}}}()
count = 0
for i in 1:2200
    if (i % 300 == 2)
        JLD.save("res.jld", "data", res)
    end
    
    if GeCo.FicoClassifier.predict(classifier, DataFrame(X[i, :]))[1] > 0.5
        continue
    end
    println(i)
    explanation,  = explain(X[i, :], X, p, classifier)
    changes = fetch_actions(explanation, X[i, :], num_actions = 3)
    push!(res, changes)
    
end

JLD.save("res.jld", "data", res)

