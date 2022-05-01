include("../src/GeneticCF.jl")
include("../scripts/fico/fico-setup.jl")

import JLD
function geco_verify(rules, orig_instance, X, classifier)
    # verify using geco
    rule = rules[1]

    # create the constrains for plaf by the rule
    p, = rule_to_constrain(rule, initPLAF(), orig_instance)

    geco_explanation, = explain(orig_instance, X, p, classifier)
    if size(geco_explanation)[1] != 0 && geco_explanation[1, :outc]
        return false
    end
    return true
end



function run_exp(X, classifier)
    plaf = initPLAF()

    preds = GeCo.score(classifier, X, 1, extra_col = 0)

    checked = 0
    index = 0

    # failed to generated rules
    fail_generated_n = 0
    fail_generated_i = 0
    fail_generated_m = 0
    fail_generated_im = 0

    # cardinality
    cardinality_n = Vector{Float64}()
    cardinality_i = Vector{Float64}()
    cardinality_m = Vector{Float64}()
    cardinality_im = Vector{Float64}()
    cardinality_all = Vector{Float64}()

    success_verified_n = 0
    success_verified_i = 0
    success_verified_m = 0
    success_verified_im = 0
    total_verified_n = 0
    total_verified_i = 0
    total_verified_m = 0
    total_verified_im = 0

    # efficiency -- runtime and generations
    times_n = Array{Float64,1}()
    times_i = Array{Float64,1}()
    times_m = Array{Float64,1}()
    times_im = Array{Float64,1}()
    times_all = Array{Float64,1}()

    generation_n = Vector{Float64}()
    generation_i = Vector{Float64}()
    generation_m = Vector{Float64}()
    generation_im = Vector{Float64}()
    generation_all = Vector{Float64}()

    while checked < 500
        index += 1

        if preds[index] >= 0.5
            continue
        end


        println(" num checked $(checked), current index$(index)")

        orig_instance = X[index, :]

        # # only geco
        # # println(orig_instance)
        # time = @elapsed results = geco_for_rule(orig_instance, X, classifier, plaf)
        # rule = results[1]
        # num_exp = results[2]

        # if num_exp == -1
        #     println("skipped")
        #     continue
        # end

        # if sum(rule.bit_representation) < 6
        #     println("find target at $(index) with car $(sum(rule.bit_representation))")
        # end

        # push!(cardinality_all, sum(rule.bit_representation))
        # push!(times_all, time)
        # push!(generation_all, num_exp)

        # if checked < 20
        #     println([sum(rule.bit_representation), time, num_exp])
        # end

        # n
        if checked < 20
            println("n")
        end
        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = false, geco_mutation = false, max_num_population=20)
        rules = results[1]
        generation = results[2]
        println(time)

        if rules[1].precision < 1
            fail_generated_n += 1
            println("fail generate for n at $(index)")
        else
            # check geco
            total_verified_n += 1
            push!(cardinality_n, sum(rules[1].bit_representation))
            push!(times_n, time)
            push!(generation_n, generation)
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_n += 1
            else
                println("geco_verify fail for n at $(index)")
            end
        end

        # i
        if checked < 20
            println("i")
        end
        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = true, geco_mutation = false, max_num_population=20)
        rules = results[1]
        generation = results[2]
        if checked < 20
            println(time)
        end

        if rules[1].precision < 1
            fail_generated_i += 1
            println("fail generate for i at $(index)")
        else
            # check geco
            total_verified_i += 1
            push!(cardinality_i, sum(rules[1].bit_representation))
            push!(times_i, time)
            push!(generation_i, generation)
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_i += 1
            else
                println("geco_verify fail for i at $(index)")
            end
        end

        # m
        if checked < 20
            println("m")
        end
        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = false, geco_mutation = true, max_num_population=10,max_generation=20,geco_check_size=10)
        rules = results[1]
        generation = results[2]
        if checked < 20
            println(time)
        end

        if rules[1].precision < 1
            fail_generated_m += 1
            println("fail generate for m at $(index)")
        else
            # check geco
            total_verified_m += 1
            push!(cardinality_m, sum(rules[1].bit_representation))
            push!(times_m, time)
            push!(generation_m, generation)
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_m += 1
            else
                println("geco_verify fail for m at $(index)")
            end
        end

        # im
        if checked < 20
            println("im")
        end
        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = true, geco_mutation = true, max_num_population=10,max_generation=30,geco_check_size=10)
        rules = results[1]
        generation = results[2]
        if checked < 20
            println(time)
        end

        if rules[1].precision < 1
            fail_generated_im += 1
            println("fail generate for im at $(index)")
        else
            # check geco
            total_verified_im += 1
            push!(cardinality_im, sum(rules[1].bit_representation))
            push!(times_im, time)
            push!(generation_im, generation)
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_im += 1
                if (sum(rules[1].bit_representation) < 6) 
                    println("find target at $(index) with car $(sum(rules[1].bit_representation))")
                end
            else
                println("geco_verify fail for im at $(index) with max generation $(generation)")

            end
        end

        checked += 1

        # if checked < 10 && length(cardinality_n) != 0 && length(cardinality_m) != 0&& length(cardinality_i) != 0&& length(cardinality_im) != 0

        #     println([fail_generated_n, fail_generated_i, fail_generated_m, fail_generated_im])
        #     println([success_verified_n, success_verified_i, success_verified_m, success_verified_im])
        #     println([cardinality_n[length(cardinality_n)], cardinality_i[length(cardinality_i)], cardinality_m[length(cardinality_m)], cardinality_im[length(cardinality_im)]])
        #     # println([cardinality_n[length(cardinality_n)], cardinality_i[length(cardinality_i)], cardinality_m[length(cardinality_m)], cardinality_im[length(cardinality_im)], cardinality_all[length(cardinality_all)]])
        # end

        if checked % 10 == 0

            fail_generates = [fail_generated_n, fail_generated_i, fail_generated_m, fail_generated_im]
            # cardinalities = [cardinality_n, cardinality_i, cardinality_m, cardinality_im, cardinality_all]
            cardinalities = [cardinality_n, cardinality_i, cardinality_m, cardinality_im]
            success_verifieds = [success_verified_n, success_verified_i, success_verified_m, success_verified_im]
            total_verifieds = [total_verified_n, total_verified_i, total_verified_m, total_verified_im]
            # times = [times_n, times_i, times_m, times_im, times_all]
            # generations = [generation_n, generation_i, generation_m, generation_im, generation_all]
            times = [times_n, times_i, times_m, times_im]
            generations = [generation_n, generation_i, generation_m, generation_im]
            
            file_naive = "rule_based_model/scripts/results/fico_$(checked).jld"
            JLD.save(file_naive, "fail_generates", fail_generates, "cardinalities", cardinalities, "total_verifieds", total_verifieds, "success_verifieds", success_verifieds, "times", times, "generations", generations)
            # file_naive = "rule_based_model/scripts/results/fico_all_$(checked).jld"
            # JLD.save(file_naive, "cardinality_all", cardinality_all, "times_all", times_all, "generation_all", generation_all)
            println("save $(checked)")
        end

    end

    fail_generates = [fail_generated_n, fail_generated_i, fail_generated_m, fail_generated_im]
    # cardinalities = [cardinality_n, cardinality_i, cardinality_m, cardinality_im, cardinality_all]
    cardinalities = [cardinality_n, cardinality_i, cardinality_m, cardinality_im]
    success_verifieds = [success_verified_n, success_verified_i, success_verified_m, success_verified_im]
    total_verifieds = [total_verified_n, total_verified_i, total_verified_m, total_verified_im]
    # times = [times_n, times_i, times_m, times_im, times_all]
    # generations = [generation_n, generation_i, generation_m, generation_im, generation_all]
    times = [times_n, times_i, times_m, times_im]
    generations = [generation_n, generation_i, generation_m, generation_im]

    # file_naive = "rule_based_model/scripts/results/credit.jld"
    file_naive = "rule_based_model/scripts/results/fico.jld"

    JLD.save(file_naive, "fail_generates", fail_generates, "cardinalities", cardinalities, "total_verifieds", total_verifieds, "success_verifieds", success_verifieds, "times", times, "generations", generations)
    # file_naive = "rule_based_model/scripts/results/fico_all.jld"
    # JLD.save(file_naive, "cardinality_all", cardinality_all, "times_all", times_all, "generation_all", generation_all)
    
    println("finished")

end
