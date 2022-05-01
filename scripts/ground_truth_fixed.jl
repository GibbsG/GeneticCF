

using GeCo, DataFrames, JLD
import StatsBase

macro ClassifierGenerator(rules, thresholds)
    return generateClassifierFunction(rules, thresholds)
end

# only if all thresholds hold (larger than or equal to the threshold), will return as 0. Otherwise, 1.
function generateClassifierFunction(rules, thresholds)
    # println("rules")
    # println(rules)
    # println(thresholds)
    clf = quote
        _instances ->
        begin
            score = Array{Float64,1}(undef, nrow(_instances))
            for i in 1:nrow(_instances)
                score[i] = 0

                # println($rules)
                # println($thresholds)
                for (feat, thresh) in zip($rules, ($thresholds))
                    if _instances[i,feat] < thresh
                        score[i] = 1
                        break
                    end
                end

            end
            score
        end
    end
    # @show clf
    esc(clf)
end

# # only if all thresholds hold (larger than or equal to the threshold), will return as 0. Otherwise, 1.
# function generateClassifierFunction(rules, thresholds)
#     clf = quote
#         _instances ->
#         begin
#             score = Array{Float64,1}(undef, nrow(_instances))
#             for i in 1:nrow(_instances)
#                 score[i] = 1

#                 for j in 1:length(rules)
#                     # rule = rules[j]
#                     # threshold = thresholds[j]

#                     failed_conditions = 1
#                     for (feat, thresh) in zip($rules[j], $thresholds[j])
#                         if _instances[i,feat] < thresh
#                             failed_conditions = 0
#                             break
#                         end
#                     end

#                     if failed_conditions == 1
#                         score[i] = 0
#                         break
#                     end
#                 end

#             end
#             score
#         end
#     end
#     # @show clf
#     esc(clf)
# end

function generatRules(vars, num)
    rules = []
    # each rule contains 1 to length(vars)/2 components
    cardinalities = rand(2:Int(floor(length(vars)/2)), num)

    for i in 1:num
        cardinility = cardinalities[i]
        features = StatsBase.sample(vars, cardinility, replace = false)

        append!(rules, [features])
    end


    return rules
end

function generateThreds(features, feature_index, rule_size, X)
    threshs = []
    rules = []

    for i in 1:2000
        orig_instance = X[i,:]

        j = 1
        thresh = []
        rule = []

        while j <= length(feature_index) && length(rule) < rule_size
            s = features[feature_index[j]]
            if orig_instance[s] != minimum(X[s])
                
                push!(rule, s)
                append!(thresh, orig_instance[s])
            end
            j += 1
        end
        append!(threshs, [thresh])
        append!(rules, [rule])
        
    end

    return threshs, rules
end

function verify_rule(expected, generated)
    num_component = 0
    for rule_component in generated.rule_components
        feature = rule_component.group.features[1]
        if rule_component.direction && feature in expected
            num_component += 1
        end
    end


    if num_component != length(expected)
        return false
    end

    return true
end



function groundTruthExperiment(X, p, thresholds, rules)

    fail_generated_n = 0
    fail_generated_im = 0    
    fail_generated_all = 0

    cardinality_n = Vector{Float64}()
    cardinality_im = Vector{Float64}()
    cardinality_all = Vector{Float64}()

    true_cardinality_n = Vector{Float64}()
    true_cardinality_im = Vector{Float64}()
    true_cardinality_all = Vector{Float64}()

    times_n = Array{Float64,1}()
    times_im = Array{Float64,1}()
    times_all = Array{Float64,1}()

    generation_n = Vector{Float64}()
    generation_im = Vector{Float64}()

    num_checked = 0

    for i in 1:length(rules)

        if num_checked >= 1000
            break
        end

        println(i)
        rule = rules[i]
        threshold = thresholds[i]
        if length(threshold) != 6
            println("skiped")
            continue
        end

        num_checked += 1

        classifier = @ClassifierGenerator(rule, threshold)
        orig_instance = X[i, :]
        
        time = @elapsed (rules_generated, generation) = generate_rules(orig_instance, X, classifier, p, geco_initial = false, geco_mutation = false);
        if verify_rule(rule, rules_generated[1])
            append!(cardinality_n, sum(rules_generated[1].bit_representation))
            append!(times_n, time)
            append!(generation_n, generation)
            append!(true_cardinality_n, length(thresholds[i]))
        else
            fail_generated_n += 1
            println("fail i")
            println(rule)
            print_rule(orig_instance, rules_generated[1])
        end

        time = @elapsed (rules_generated, generation) = generate_rules(orig_instance, X, classifier, p, geco_initial = true, geco_mutation = true);
        if verify_rule(rule, rules_generated[1])
            append!(cardinality_im, sum(rules_generated[1].bit_representation))
            append!(times_im, time)
            append!(generation_im, generation)
            append!(true_cardinality_im, length(thresholds[i]))
        else
            fail_generated_im += 1
            println("fail im")
            println(rule)
            print_rule(orig_instance, rules_generated[1])
        end

        time = @elapsed results = geco_for_rule(orig_instance, X, classifier, p)
        if verify_rule(rule, results[1])
            append!(cardinality_all, sum(results[1].bit_representation))
            append!(times_all, time)
            append!(true_cardinality_all, length(thresholds[i]))
        else
            println("fail all")
            println(rule)
            print_rule(orig_instance, results[1])
            fail_generated_all += 1
        end

        if i % 10 == 5
            println(i)
        end

    end

    fail_generates = [fail_generated_n, fail_generated_im, fail_generated_all]
    cardinalities = [cardinality_n, cardinality_im, cardinality_all]
    true_cardinalities = [true_cardinality_n, true_cardinality_im, true_cardinality_all]
    times = [times_n, times_im, times_all]
    generations = [generation_n, generation_im]
    
    file_naive = "rule_based_model/scripts/results/ground_truth_credit_6_2.jld"

    JLD.save(file_naive, "fail_generates", fail_generates, "cardinalities", cardinalities, "true_cardinalities", true_cardinalities, "times", times, "generations", generations)

    println("finished")

end



include("../../scripts/credit/credit_setup_MACE.jl");
include("../../rule_based_model/src/genetic_rule.jl");

vars = [
    :MaxBillAmountOverLast6Months,
    :MostRecentBillAmount,
    :MaxPaymentAmountOverLast6Months,
    :MostRecentPaymentAmount,
    :TotalMonthsOverdue,
    :MonthsWithZeroBalanceOverLast6Months,
    :MonthsWithLowSpendingOverLast6Months,
    :MonthsWithHighSpendingOverLast6Months,
    :AgeGroup,
    :EducationLevel,
    :TotalOverdueCounts,
    :HasHistoryOfOverduePayments
]

features = [11, 4, 9, 5, 6, 3, 2, 1, 10, 0, 7, 8] .+ 1
threds, rules = generateThreds(vars, features, 6, X)

groundTruthExperiment(X, initPLAF(), threds, rules)