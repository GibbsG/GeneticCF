include("../src/GeneticCF.jl")
include("../scripts/credit/credit_setup_MACE.jl")
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

    while checked < 3
        index += 1

        if preds[index] >= 0.5
            continue
        end

        n_success = false
        i_success = false
        m_success = false
        im_success = false


        println(" num checked $(checked), current index$(index)")

        orig_instance = X[index, :]

        # only geco
        # println(orig_instance)
        time = @elapsed results = geco_for_rule(orig_instance, X, classifier, plaf)
     
        rule = results[1]
        num_exp = results[2]

        if num_exp == -1
            println("skipped")
            continue
        end

        push!(cardinality_all, sum(rule.bit_representation))
        push!(times_all, time)
        push!(generation_all, num_exp)

        # n

        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = false, geco_mutation = false)
        rules = results[1]
        generation = results[2]

        if rules[1].precision < 1
            fail_generated_n += 1
            println("fail generate for n at $(index)")
        else
            # check geco
            total_verified_n += 1
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_n += 1
                push!(cardinality_n, sum(rules[1].bit_representation))
                push!(times_n, time)
                push!(generation_n, generation)
                n_success = true
            else
                println("geco_verify fail for n at $(index)")
            end
        end

        # i

        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = true, geco_mutation = false)
        rules = results[1]
        generation = results[2]


        if rules[1].precision < 1
            fail_generated_i += 1
            println("fail generate for i at $(index)")
        else
            # check geco
            total_verified_i += 1
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_i += 1
                push!(cardinality_i, sum(rules[1].bit_representation))
                push!(times_i, time)
                push!(generation_i, generation)
                i_success = true
            else
                println("geco_verify fail for i at $(index)")
            end
        end

        # m

        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = false, geco_mutation = true)
        rules = results[1]
        generation = results[2]

        if rules[1].precision < 1
            fail_generated_m += 1
            println("fail generate for m at $(index)")
        else
            # check geco
            total_verified_m += 1
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_m += 1
                push!(cardinality_m, sum(rules[1].bit_representation))
                push!(times_m, time)
                push!(generation_m, generation)
                m_success = true
            else
                println("geco_verify fail for m at $(index)")
            end
        end

        # im
        time = @elapsed results = generate_rules(orig_instance, X, classifier, plaf, geco_initial = true, geco_mutation = true)
        rules = results[1]
        generation = results[2]

        if rules[1].precision < 1
            fail_generated_im += 1
            println("fail generate for im at $(index)")
        else
            # check geco
            total_verified_im += 1
            if geco_verify(rules, orig_instance, X, classifier)
                success_verified_im += 1
                push!(cardinality_im, sum(rules[1].bit_representation))
                push!(times_im, time)
                push!(generation_im, generation)
                im_success = true
            else
                println("geco_verify fail for im at $(index)")
            end
        end

        checked += 1

        if !(n_success && i_success && m_success && im_success)
            # some failed
            if n_success
                deleteat!(cardinality_n, length(cardinality_n))
                deleteat!(times_n, length(times_n))
                deleteat!(generation_n, length(generation_n))
            end

            if i_success
                deleteat!(cardinality_i, length(cardinality_i))
                deleteat!(times_i, length(times_i))
                deleteat!(generation_i, length(generation_i))
            end

            if m_success
                deleteat!(cardinality_m, length(cardinality_m))
                deleteat!(times_m, length(times_m))
                deleteat!(generation_m, length(generation_m))
            end

            if im_success
                deleteat!(cardinality_im, length(cardinality_im))
                deleteat!(times_im, length(times_im))
                deleteat!(generation_im, length(generation_im))
            end

            deleteat!(cardinality_all, length(cardinality_all))
            deleteat!(times_all, length(times_all))
            deleteat!(generation_all, length(generation_all))
        elseif checked < 10

            println([fail_generated_n, fail_generated_i, fail_generated_m, fail_generated_im])
            println([success_verified_n, success_verified_i, success_verified_m, success_verified_im])
            println([cardinality_n[length(cardinality_n)], cardinality_i[length(cardinality_i)], cardinality_m[length(cardinality_m)], cardinality_im[length(cardinality_im)], cardinality_all[length(cardinality_all)]])
        end

        if checked % 10 == 0

            fail_generates = [fail_generated_n, fail_generated_i, fail_generated_m, fail_generated_im]
            cardinalities = [cardinality_n, cardinality_i, cardinality_m, cardinality_im, cardinality_all]
            success_verifieds = [success_verified_n, success_verified_i, success_verified_m, success_verified_im]
            total_verifieds = [total_verified_n, total_verified_i, total_verified_m, total_verified_im]
            times = [times_n, times_i, times_m, times_im, times_all]
            generations = [generation_n, generation_i, generation_m, generation_im, generation_all]
            
            file_naive = "rule_based_model/scripts/results/credit_$(checked).jld"

            JLD.save(file_naive, "fail_generates", fail_generates, "cardinalities", cardinalities, "total_verifieds", total_verifieds, "success_verifieds", success_verifieds, "times", times, "generations", generations)

            println("save $(checked)")
        end

    end

    fail_generates = [fail_generated_n, fail_generated_i, fail_generated_m, fail_generated_im]
    cardinalities = [cardinality_n, cardinality_i, cardinality_m, cardinality_im, cardinality_all]
    success_verifieds = [success_verified_n, success_verified_i, success_verified_m, success_verified_im]
    total_verifieds = [total_verified_n, total_verified_i, total_verified_m, total_verified_im]
    times = [times_n, times_i, times_m, times_im, times_all]
    generations = [generation_n, generation_i, generation_m, generation_im, generation_all]
    
    file_naive = "/scripts/results/credit.jld"

    JLD.save(file_naive, "fail_generates", fail_generates, "cardinalities", cardinalities, "total_verifieds", total_verifieds, "success_verifieds", success_verifieds, "times", times, "generations", generations)

    println("finished")

end
