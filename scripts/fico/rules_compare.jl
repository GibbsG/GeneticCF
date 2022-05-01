using JLD
using Pickle

GeCo_res = JLD.load("/Users/gzx/Desktop/research/geco/GeCo.jl/res.jld")["data"]
rules_res = Pickle.load(open( "/Users/gzx/Desktop/research/setCover/GloballyConsistentRules/rules.pickle"))

corrects_val = 0
counts_val = 0

# for entity_index in 1:length(GeCo_res)
for entity_index in 1:180
    rule = rules_res[entity_index]
    geco = GeCo_res[entity_index]
    for expl_index in 1:length(geco)
        global counts_val += 1
        geco_expl = geco[expl_index]
        contains = false
        for feature_index in 1:length(geco_expl)
            print()
            if haskey(rule, geco_expl[feature_index][1]) && (rule[geco_expl[feature_index][1]] == geco_expl[feature_index][2] || rule[geco_expl[feature_index][1]] == 3)
                contains = true
                break
            end
        end
        if contains
            global corrects_val += 1
        end
    end 
end

println("Total number of explanations: $(counts_val)")
println("Total number of explanations following the rule: $(corrects_val)")
println("The fraction of explanations following the rule: $(corrects_val / counts_val)")
println("The fraction of explanations not following the rule: $(1 - corrects_val / counts_val)")