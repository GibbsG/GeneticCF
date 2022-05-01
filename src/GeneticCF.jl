
using DataFrames, Statistics, MLJScientificTypes
import StatsBase

# the feature group -- help for one-hot encodings and also for categorifal feature
# a feature must be in exactly one group
struct RuleFeatureGroup
    features::Vector{Symbol}
    indexes::Vector{Int32}
    categorical::Bool
end

function initRuleFeatureGroups(prog::PLAFProgram, data::DataFrame)
    groups = Array{RuleFeatureGroup, 1}()
    fidx_groupidx = Array{Int32}(undef, size(data)[2])

    includedFeatures = Set{Symbol}()

    for group in prog.groups
        indexes = Vector{Int32}()
        for (idx, feature) in enumerate(group)
            @assert feature ∈ propertynames(data) "The feature $feature does not occur in the input data."
            @assert feature ∉ includedFeatures "Each feature can be in at most one group!"
            push!(includedFeatures, feature)
            push!(indexes, findfirst(isequal(feature),propertynames(data)))
        end

        push!(groups,
            RuleFeatureGroup(
                [group...],
                indexes,
                true
                )
            )
    end

    for feature in propertynames(data)
        if feature ∉ includedFeatures
            @assert feature ∈ propertynames(data) "The feature $feature does not occur in the input data."
            indexes = [findfirst(isequal(feature),propertynames(data))]

            push!(groups,
                RuleFeatureGroup(
                    [feature],
                    indexes,
                    elscitype(data[!,feature]) == Multiclass
                    )
            )
        end
    end

    for (gidx, group) in enumerate(groups)
        for fidx in group.indexes
            fidx_groupidx[fidx] = gidx
        end
    end

    return groups, fidx_groupidx
end


struct RuleComponent
    group::RuleFeatureGroup
    direction::Bool
    index::Int64
    group_index::Int64
end

function sort_components(component::RuleComponent)
    return component.index
end

function initailComponents(groups::Array{RuleFeatureGroup, 1})
    components = Array{RuleComponent, 1}()
    for (idx, group) in enumerate(groups)
        push!(components, RuleComponent(group, true, 2*idx-1, idx))
        push!(components, RuleComponent(group, false, 2*idx, idx))
    end
    return components
end


# the rule generated
mutable struct Rule
    rule_components::Array{RuleComponent, 1} # all components of this rule
    bit_representation::Array{Bool, 1} # check whether two rule are the same
    score::Float64
    precision::Float64
    generation::Int64
    geco_verified::Bool
    geco_generated::Bool
end

function initialRule(component_num)::Rule
    return Rule(Array{RuleComponent, 1}(), falses(component_num), 0, 0, 0, 0, 0)
end

function print_rule(orig_instance::DataFrameRow, rule::Rule)
    println("The precision of the rule is: $(rule.precision)")
    println("This rule has $(sum(length(rule.rule_components))) components")

    included_groups = Set{RuleFeatureGroup}()
    index = 1

    for rule_component in rule.rule_components
        if rule_component.group in included_groups
            # this group has already been printed
            continue
        end

        # check whether the other direction also in the group
        bidirection = false
        for rule_component_other in rule.rule_components
            if rule_component_other.group_index == rule_component.group_index && rule_component_other.direction != rule_component.direction
                bidirection = true
                break
            end
        end

        if bidirection
            if rule_component.direction
                continue
            end
            println("\t Component $(index) and $(index + 1) defines exact match on the feature group:")
            index += 1
        elseif rule_component.direction # >=
            println("\t Component $(index) defines lower bound on the feature group:")
        else
            println("\t Component $(index) defines upper bound on the feature group:")
        end

        for feature in rule_component.group.features
            if bidirection
                println("\t\t For feature $(feature): x = $(orig_instance[feature])")
            elseif rule_component.direction # >=
                println("\t\t For feature $(feature): x >= $(orig_instance[feature])")
            else
                println("\t\t For feature $(feature): x <= $(orig_instance[feature])")
            end
        end

        index += 1
        push!(included_groups, rule_component.group)
    end

end

struct ActionComponent
    index::Int32
    direction::Int32 # 1 -> become larger
end

struct Action
    components::Vector{ActionComponent}
end

function sort_action_cardinality(action)
    return length(action.components)
end  

function generate_rules(orig_instance::DataFrameRow, data::DataFrame, classifier, plaf_program::PLAFProgram; 
    max_num_population::Int64 = 50,
    geco_initial::Bool = false,
    geco_mutation::Bool = false,
    desired_class::Int64 = 1,
    max_generation::Int64 = 50,
    sample_num::Int64 = 1000,
    geco_check_size::Int64 = 5,
    ablation::Bool = false)

    epslin = 0.0
    if !ablation
        groups::Array{RuleFeatureGroup, 1}, fidx_gidx::Vector{Int32} = initRuleFeatureGroups(plaf_program, data)
        components::Array{RuleComponent, 1} = initailComponents(groups)

        population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}} = [], Set()

        population, checked_set = initialPopulation_naive(components)

        if geco_initial
            # print("geco Init")
            # @time population, checked_set = initialPopulationWithGeco(data, orig_instance, plaf_program, classifier, fidx_gidx, components, population, checked_set, desired_class = desired_class)
            population, checked_set = initialPopulationWithGeco(data, orig_instance, plaf_program, classifier, fidx_gidx, components, population, checked_set, desired_class = desired_class)
        end

        space = initialSpace(orig_instance, data)
        selection!(population, data, orig_instance, space, classifier, checked_set, pop_size = max_num_population, epslin = epslin, desired_class = desired_class, sample_num = sample_num)
        
        generation = 0
        converge = false

        geco_checked = true
        if epslin == 0.0 && geco_mutation
            geco_checked = false
        end

        last_geco = 0

        while generation < max_generation && (!converge || length(population) < 1 || population[1].precision < 1 - epslin) 
            generation += 1
            # println(generation)
            # println(length(population))

            if geco_mutation && generation - last_geco > 3
                last_geco = generation
                # print("geco mutation")
                # @time mutation_geco!(population, checked_set, data, orig_instance, classifier, plaf_program, fidx_gidx, components, generation, desired_class = desired_class, check_size = geco_check_size)
                mutation_geco!(population, checked_set, data, orig_instance, classifier, plaf_program, fidx_gidx, components, generation, desired_class = desired_class, check_size = geco_check_size)
            end

            ori_pop_size = length(population)

            crossover!(population, checked_set, components, generation, pop_size = max_num_population)
            mutation!(population, checked_set, components, generation, pop_size = min(ori_pop_size, max_generation))
        
            selection!(population, data, orig_instance, space, classifier, checked_set, pop_size = max_num_population, epslin = epslin, desired_class = desired_class, sample_num = sample_num)

            converge = check_converge(population, generation, geco_mutation)
            if geco_mutation && converge
                if population[1].geco_verified
                    population[1].geco_verified = false
                    changed = mutation_geco!(population, checked_set, data, orig_instance, classifier, plaf_program, fidx_gidx, components, generation, desired_class = desired_class, check_size = 1)
                    
                    selection!(population, data, orig_instance, space, classifier, checked_set, pop_size = max_num_population, epslin = epslin, desired_class = desired_class, sample_num = sample_num)
                    if changed
                        converge = false
                    end

                else
                    converge = false
                end
            end
        
        end

        return population, generation
    else
        prep_time = 0.0
        selection_time = 0.0
        mutation_time = 0.0
        crossover_time = 0.0
        geco_init_time = 0.0
        geco_mutation_time = 0.0
        num_explored = 0
        num_explored_geco = 0

        ptime = @elapsed groups, fidx_gidx = initRuleFeatureGroups(plaf_program, data)
        components = initailComponents(groups)
        prep_time += ptime

        population, checked_set = [], Set()

        ptime = @elapsed  population, checked_set = initialPopulation_naive(components)
        prep_time += ptime

        if geco_initial
            # print("geco Init")
            # @time population, checked_set = initialPopulationWithGeco(data, orig_instance, plaf_program, classifier, fidx_gidx, components, population, checked_set, desired_class = desired_class)
            gitime = @elapsed population, checked_set = initialPopulationWithGeco(data, orig_instance, plaf_program, classifier, fidx_gidx, components, population, checked_set, desired_class = desired_class)
            geco_init_time = gitime
            num_explored_geco += 1
        end

        ptime = @elapsed space = initialSpace(orig_instance, data)
        prep_time += ptime

        num_explored += max(0, length(population) - max_num_population)
        stime = @elapsed selection!(population, data, orig_instance, space, classifier, checked_set, pop_size = max_num_population, epslin = epslin, desired_class = desired_class, sample_num = sample_num)
        selection_time += stime

        generation = 0
        converge = false

        geco_checked = true
        if epslin == 0.0 && geco_mutation
            geco_checked = false
        end

        last_geco = 0

        while generation < max_generation && (!converge || length(population) < 1 || population[1].precision < 1 - epslin) 
            generation += 1
            # println(generation)
            # println(length(population))

            if geco_mutation && generation - last_geco > 3
                last_geco = generation
                # print("geco mutation")
                # @time mutation_geco!(population, checked_set, data, orig_instance, classifier, plaf_program, fidx_gidx, components, generation, desired_class = desired_class, check_size = geco_check_size)
                num_explored_geco += min(length(population), geco_check_size)
                gmtime = @elapsed mutation_geco!(population, checked_set, data, orig_instance, classifier, plaf_program, fidx_gidx, components, generation, desired_class = desired_class, check_size = geco_check_size)
                geco_mutation_time += gmtime
            end

            ori_pop_size = length(population)

            ctime = @elapsed crossover!(population, checked_set, components, generation, pop_size = max_num_population)
            crossover_time += ctime

            mtime = @elapsed mutation!(population, checked_set, components, generation, pop_size = min(ori_pop_size, max_generation))
            mutation_time += mtime

            num_explored += max(0, length(population) - max_num_population)
            stime = @elapsed selection!(population, data, orig_instance, space, classifier, checked_set, pop_size = max_num_population, epslin = epslin, desired_class = desired_class, sample_num = sample_num)
            selection_time += stime

            converge = check_converge(population, generation, geco_mutation)
            if geco_mutation && converge
                if population[1].geco_verified
                    population[1].geco_verified = false
                    gmtime = @elapsed changed = mutation_geco!(population, checked_set, data, orig_instance, classifier, plaf_program, fidx_gidx, components, generation, desired_class = desired_class, check_size = 1)
                    geco_mutation_time += gmtime
                    
                    num_explored += max(0, length(population) - max_num_population)
                    stime = @elapsed selection!(population, data, orig_instance, space, classifier, checked_set, pop_size = max_num_population, epslin = epslin, desired_class = desired_class, sample_num = sample_num)
                    selection_time += stime
                    if changed
                        converge = false
                    end

                else
                    converge = false
                end

            end
        
        end

        return population, generation, num_explored + num_explored_geco, num_explored_geco, prep_time, selection_time, mutation_time, crossover_time, geco_init_time, geco_mutation_time
    end
end

function initialPopulation_naive(components::Array{RuleComponent, 1})
    # no geco facilatate

    initial_pop::Array{Rule, 1} = Vector{Rule}()
    population_set::Set{Array{Bool, 1}} = Set{Array{Bool, 1}}() # to make sure we only have unique rules

    for (index, component) in enumerate(components)
        rule = initialRule(length(components))
        push!(rule.rule_components, component)
        rule.bit_representation[index] = true
        if component.group.categorical
            if component.direction
                push!(rule.rule_components, components[index+1])
                rule.bit_representation[index+1] = true
            else
                continue
            end  
        end
        push!(initial_pop, rule)
        push!(population_set, rule.bit_representation)
    end


    return initial_pop, population_set
end

function initialSpace(orig_instance::DataFrameRow, data::DataFrame)
    space = Vector{Vector{Vector{Float64}}}()

    for feature in names(data)
        feature_space = Vector{Vector{Float64}}()
        all_values = data[:,feature]
        push!(feature_space, all_values[all_values .>= orig_instance[feature]])
        push!(feature_space, all_values[all_values .<= orig_instance[feature]])
        push!(space, feature_space)
    end

    return space
end

function sort_rule(rule::Rule)
    return rule.score
end


function selection!(population::Array{Rule, 1}, data::DataFrame, orig_instance::DataFrameRow, space, classifier, checked_set::Set{Array{Bool, 1}}; pop_size::Int64 = 10, epslin = 0.95, desired_class = 1, sample_num = 1000) # TODO
    # no geco facilatate
    # how to know what we should evaluate, do we need to evaluate all?
    # A: we first evaluate all
    # println(sample_num)

    for i in 1:length(population)
        if population[i].score != 0
            continue
        end

        evaluation(data, population[i], orig_instance, space, classifier, epslin = epslin, desired_class = desired_class, sample_num = sample_num)
        # println(population[i].score)
    end


    sort!(population, by = sort_rule, rev = true)
    if length(population) > pop_size
        for i in pop_size + 1:length(population)
            if population[i].score < 0.5
                break
            else
                delete!(checked_set, population[i].bit_representation)
            end
        end 
        deleteat!(population, pop_size + 1: length(population))
    end

end


function mutation!(population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}}, components::Array{RuleComponent, 1}, generation::Int64; max_num_samples::Int64 = 3, pop_size = 20)
    row_num = min(length(population), pop_size)
    col_num = length(components)

    for row_index in 1:row_num
        rule::Rule = population[row_index]

        possibel_components = []
        for i in 1:col_num
            if !rule.bit_representation[i]
                push!(possibel_components, i)
            end
        end

        changes = StatsBase.sample(possibel_components, min(length(possibel_components), max_num_samples), replace = false)
        for index in changes
            mutated_rule::Rule = deepcopy(rule)
            mutated_rule.generation = generation
            mutated_rule.geco_verified = false
            mutated_rule.geco_generated = false
            mutated_rule.score = 0

            # check whether the group is categorical
            if components[index].group.categorical
                group_index = Int(ceil(index/2))
                mutated_rule.bit_representation[group_index * 2 - 1] = true
                push!(mutated_rule.rule_components, components[group_index * 2 - 1])
                mutated_rule.bit_representation[group_index * 2] = true
                push!(mutated_rule.rule_components, components[group_index * 2])
            else
                mutated_rule.bit_representation[index] = true
                push!(mutated_rule.rule_components, components[index])
            end

            if !(mutated_rule.bit_representation in checked_set)
                sort!(mutated_rule.rule_components, by = sort_components, rev = false)
                push!(population, mutated_rule)
                push!(checked_set, mutated_rule.bit_representation)
            end
        
        end
    end
end

function crossover!(population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}}, components::Array{RuleComponent, 1}, generation::Int64; pop_size = 20)  
    row_num = min(length(population), pop_size)
    col_num = length(components)
    for parent1_index in 1:row_num
        for parent2_index in (parent1_index+1):row_num

            mutated_1::Rule = deepcopy(population[parent1_index])
            mutated_1.score = 0
            mutated_1.generation = generation
            mutated_1.geco_verified = false
            mutated_1.geco_generated = false

            mutated_2::Rule = deepcopy(population[parent2_index])
            mutated_2.generation = generation
            mutated_2.geco_verified = false
            mutated_2.geco_generated = false
            mutated_2.score = 0

            choice_1 = Vector{Int16}()
            choice_2 = Vector{Int16}()

            for i in 1:col_num
                if mutated_1.bit_representation[i] && !mutated_2.bit_representation[i]
                    push!(choice_2, i)
                elseif mutated_2.bit_representation[i] && !mutated_1.bit_representation[i]
                    push!(choice_1, i)
                end
            end


            if length(choice_1) != 0
                add_component = StatsBase.sample(choice_1)

                # check whether this is categorical
                if components[add_component].group.categorical
                    group_index = Int(ceil(add_component/2))
                    mutated_1.bit_representation[group_index * 2 - 1] = true
                    push!(mutated_1.rule_components, components[group_index * 2 - 1])
                    mutated_1.bit_representation[group_index * 2] = true
                    push!(mutated_1.rule_components, components[group_index * 2])
                else
                    mutated_1.bit_representation[add_component] = true
                    push!(mutated_1.rule_components, components[add_component])
                end

                if !(mutated_1.bit_representation in checked_set)
                    sort!(mutated_1.rule_components, by = sort_components, rev = false)
                    push!(population, mutated_1)
                    push!(checked_set, mutated_1.bit_representation)
                end
            end

            if length(choice_2) != 0
                add_component = StatsBase.sample(choice_2)
                
                if components[add_component].group.categorical
                    group_index = Int(ceil(add_component/2))
                    mutated_2.bit_representation[group_index * 2 - 1] = true
                    push!(mutated_2.rule_components, components[group_index * 2 - 1])
                    mutated_2.bit_representation[group_index * 2] = true
                    push!(mutated_2.rule_components, components[group_index * 2])
                else
                    mutated_2.bit_representation[add_component] = true
                    push!(mutated_2.rule_components, components[add_component])
                end

                if !(mutated_2.bit_representation in checked_set)
                    sort!(mutated_2.rule_components, by = sort_components, rev = false)
                    push!(population, mutated_2)
                    push!(checked_set, mutated_2.bit_representation)
                end
            end


        end
    end
end


function evaluation(data::DataFrame, rule::Rule, orig_instance::DataFrameRow, space, classifier; epslin = 0.05, desired_class = 1, sample_num = 1000)
    samples = sample_data(data, rule, orig_instance, space, number = sample_num)

    preds::Vector{Float64} = GeCo.score(classifier, samples, desired_class, extra_col = 0)

    precision = count(i->(i<0.5), preds) / length(preds)

    rule.precision = precision
    if rule.geco_verified
        rule.score = 1.0 / length(rule.bit_representation) * 0.25 + 0.5 - sum(rule.bit_representation) / length(rule.bit_representation) * 0.5 + precision * 0.5
    elseif precision < 1.0 - epslin
        if precision < 0.25 && rule.geco_generated
            precision += sum(rule.bit_representation) / length(rule.bit_representation)
        end
        rule.score = precision * 0.5
    else
        rule.score = 0.5 - sum(rule.bit_representation) / length(rule.bit_representation) * 0.5 + precision * 0.5
    end
end

function sample_data(data::DataFrame, rule::Rule, orig_instance::DataFrameRow, space; number::Int64 = 1000)
    rows = rand(1:nrow(data), number)
    
    sample_data = data[rows, :]

    for (idx, component::RuleComponent) in enumerate(rule.rule_components)
        direction = 0 # 0 means >=, 1 meas <=, 2 means =
        if idx != 1 && (component.group_index == rule.rule_components[idx-1].group_index)
            continue
        end

        if idx != length(rule.rule_components) && component.group_index == rule.rule_components[idx+1].group_index
            # exact same
            direction = 2
        elseif !component.direction
            direction = 1
        end

        for (fidx::Int32, feature::Symbol) in zip(component.group.indexes, component.group.features)
            if direction == 2 # == 
                sample_data[feature] = orig_instance[feature]
            elseif direction == 1 # <=
                violates_num = sum(sample_data[feature] .> orig_instance[feature])
                replaces = rand(1:length(space[fidx][2]), violates_num)

                sample_data[sample_data[feature] .> orig_instance[feature], feature] = space[fidx][2][replaces]
            else # >=
                violates_num = sum(sample_data[feature] .< orig_instance[feature])
                replaces = rand(1:length(space[fidx][1]), violates_num)
                sample_data[sample_data[feature] .< orig_instance[feature], feature] = space[fidx][1][replaces]
            end
        end
    end

    return sample_data
end


function initialPopulationWithGeco(data::DataFrame, orig_instance::DataFrameRow, plaf_program::PLAFProgram, classifier, fidx_gidx, components, population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}}; desired_class = 1, num_actions = 10)

    explains,  = GeCo.explain(orig_instance, data, plaf_program, classifier, desired_class = desired_class)
    
    feature_actions::Vector{Action} = fetch_feature_actions(explains, orig_instance)
    
    group_actions::Vector{Action} = actions_reduced_groups(feature_actions, fidx_gidx)

    action_components::Vector{ActionComponent} = Vector{ActionComponent}()

    backtrace_geco(group_actions, 1, action_components, components, population, checked_set, 0)
    return population, checked_set
end

function backtrace_geco(actions::Vector{Action}, index::Int64, action_components::Vector{ActionComponent}, components::Array{RuleComponent, 1}, population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}}, generation::Int64; max_action = 5)
    if index > min(length(actions), max_action)
        rule = action_to_rule(action_components, components, generation) # build bit representation and handle categorical groups
        if !(rule.bit_representation in checked_set)
            sort!(rule.rule_components, by = sort_components, rev = false)
            push!(population, rule)
            push!(checked_set, rule.bit_representation)
        end
        return
    end

    for actionComponent::ActionComponent in actions[index].components
        push!(action_components, actionComponent)
        backtrace_geco(actions, index + 1, action_components, components, population, checked_set, generation)
        pop!(action_components)
        # println(action_components)
    end
end

function action_to_rule(action_components::Vector{ActionComponent}, components::Array{RuleComponent, 1}, generation)
    rule::Rule = initialRule(length(components))
    rule.generation = generation

    for action_component in action_components
        cindex = action_component.index * 2

        if components[cindex].group.categorical
            # we should add both
            if rule.bit_representation[cindex]
                continue
            end

            # add both component for this group
            rule.bit_representation[cindex] = true
            rule.bit_representation[cindex - 1] = true
            push!(rule.rule_components, components[cindex])
            push!(rule.rule_components, components[cindex - 1])
        else
            if action_component.direction == 1
                cindex -= 1
            end

            if rule.bit_representation[cindex]
                continue
            end

            if rule.bit_representation[cindex]
                continue
            end

            # add both component for this group
            rule.bit_representation[cindex] = true
            push!(rule.rule_components, components[cindex])

        end

    end

    rule.geco_generated = true

    return rule
end

function fetch_feature_actions(counterfactuals::DataFrame, orig_instance)::Vector{Action}
    res::Vector{Action} = Vector{Action}()
    idx = 1
    while idx <= size(counterfactuals)[1] && counterfactuals[idx,:outc]
        action = Action(Vector{ActionComponent}())
        cf = counterfactuals[idx,:]
        for (fidx, feature) in enumerate(propertynames(orig_instance))
            delta = cf[feature] - orig_instance[feature]
            if delta != 0
                dir = 2  # 2 as become larger
                if delta < 0
                    dir = 1 # 1 as become smaller
                end
                push!(action.components, ActionComponent(fidx, dir))
            end
        end
        push!(res, action)
        idx += 1
    end
    return res
end

function actions_reduced_groups(feature_actions::Vector{Action}, fidx_gidx::Vector{Int32})::Vector{Action}
    group_actions = Vector{Action}()
    # convert feature actions to group actions
    for feature_action::Action in feature_actions
        action = Action(Vector{ActionComponent}())
        group_indexes = Vector{Int32}()
        for action_component in feature_action.components
            fidx = action_component.index
            dir = action_component.direction
            gidx = fidx_gidx[fidx]
            if gidx in group_indexes
                continue
            end
            push!(group_indexes, gidx)
            push!(action.components, ActionComponent(gidx, dir))
            # print!(action.components)
        end
        push!(group_actions, action)
    end

    # println(group_actions)

    # reduce the group actions
    sort!(group_actions, by = sort_action_cardinality)
    
    aidx = 2
    while aidx <= length(group_actions)
        # check whether we can remove this group
        cur_action = group_actions[aidx]
        covered = true
        for prev_index in 1:aidx-1
            covered = true
            prev_action = group_actions[prev_index]
            for action_component in prev_action.components
                if ! (action_component in cur_action.components)
                    covered = false
                    break
                end
            end

            if covered
                break
            end
        end

        if covered 
            deleteat!(group_actions, aidx)
        else
            aidx += 1
        end
    end

    return group_actions
end
           

function mutation_geco!(population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}}, data::DataFrame, orig_instance::DataFrameRow, classifier, plaf_program::PLAFProgram, fidx_gidx, components, generation; desired_class = 1, check_size = 10)
    row_num = min(length(population), check_size)

    idx = 1

    changed = false

    while row_num != 0
        row_num -= 1

        rule = population[idx]
        if rule.geco_verified
            idx += 1
            continue
        end

        p, action_components = rule_to_constrain(rule, plaf_program, orig_instance)

        # print("explain")
        explains, = explain(orig_instance, data, p, classifier, desired_class = desired_class)
        if size(explains)[1] == 0 || !explains[1,:outc]
            rule.geco_verified = true
            idx += 1
            continue
        end

        rule.geco_verified = false
        cur_pop_size = length(population)
        changed = true

        feature_actions::Vector{Action} = fetch_feature_actions(explains, orig_instance)
        
        group_actions::Vector{Action} = actions_reduced_groups(feature_actions, fidx_gidx)

        backtrace_geco(group_actions, 1, action_components, components, population, checked_set, generation)
        
        if cur_pop_size < length(population) || cur_pop_size > 2
            deleteat!(population, idx)
        end
    end
    return changed
end

function rule_to_constrain(rule::Rule, plaf_program::PLAFProgram, orig_instance::DataFrameRow)
    plaf = PLAFProgram(plaf_program)
    current_components::Vector{ActionComponent} = Vector{ActionComponent}()
    for (idx, rule_component) in enumerate(rule.rule_components)

        group_index = Int(ceil(rule_component.index/2))
        
        direction = 0 # 0 means >=, 1 meas <=, 2 means =
        if idx != 1 && (rule_component.group_index == rule.rule_components[idx-1].group_index)
            continue
        end

        if idx != length(rule.rule_components) && rule_component.group_index == rule.rule_components[idx+1].group_index
            # exact same
            direction = 2
            push!(current_components, ActionComponent(group_index, 1))
            push!(current_components, ActionComponent(group_index, 2))
        elseif rule_component.direction
            push!(current_components, ActionComponent(group_index, 1))
        else
            direction = 1
            push!(current_components, ActionComponent(group_index, 2))
        end

        # add constrain for each feature in the group
        for (fidx, feature) in zip(rule_component.group.indexes, rule_component.group.features)
            c = Int64(orig_instance[feature])
            if direction == 2 # == 
                plaf = @PLAF(plaf, :cf.aa .== parse(Int64, "$c"))
            elseif direction == 1 # <=
                plaf = @PLAF(plaf, :cf.aa .<= parse(Int64, "$c"))
            else # >=
                plaf = @PLAF(plaf, :cf.aa .>= parse(Int64, "$c"))
            end

            cur_index = length(plaf.constraints)
            push!(plaf.constraints[cur_index].features, feature)
            deleteat!(plaf.constraints[cur_index].features, 1)
        end

    end
    return plaf, current_components
end


function check_converge(population::Array{Rule, 1}, generation::Int64, geco_mutation)
    
    if geco_mutation && population[1].geco_verified && population[1].generation + 3 <= generation
        return true
    end

    i = 1

    for rule in population
        if rule.generation + 1 >= generation
            return false
        end


        if i >= 3 && population[1].geco_verified
            return true
        else
            if i >= 5
                return true
            end
        end

        i += 1
    end

    return true
end

function greedyCF(orig_instance::DataFrameRow, data::DataFrame, classifier, plaf_program::PLAFProgram; desired_class = 1)

    groups::Array{RuleFeatureGroup, 1}, fidx_gidx::Vector{Int32} = initRuleFeatureGroups(plaf_program, data)
    components::Array{RuleComponent, 1} = initailComponents(groups)

    initial_pop::Array{Rule, 1} = Vector{Rule}()
    population_set::Set{Array{Bool, 1}} = Set{Array{Bool, 1}}()
    population::Array{Rule, 1}, checked_set::Set{Array{Bool, 1}} = initialPopulationWithGeco(data, orig_instance, plaf_program, classifier, fidx_gidx, components, initial_pop, population_set, desired_class = desired_class)
    
    sort!(population, by = sort_rule_cardinality, rev = true)
    num_exp = 1

    while true
        rule::Rule = pop!(population)
        num_exp += 1
        p_cur, action_components = rule_to_constrain(rule, plaf_program,orig_instance)
        explains, = explain(orig_instance, data, p_cur, classifier, desired_class = desired_class)
        if size(explains)[1] == 0 || !explains[1, :outc]
            return rule, num_exp
        end
        
        # we should create new rules based on this rule and explains
        feature_actions::Vector{Action} = fetch_feature_actions(explains, orig_instance)
        group_actions::Vector{Action} = actions_reduced_groups(feature_actions, fidx_gidx)

        backtrace_geco(group_actions, 1, action_components, components, population, checked_set, 0)

        if length(population) == 0
            return rule, -1
        end
        
        sort!(population, by = sort_rule_cardinality, rev = true)
    end
end



function sort_rule_cardinality(rule::Rule)
    return length(rule.rule_components)
end