# GeneticCF: Using Counterfactual Explanations in Generating Rule-based Explanations

To run GeneticCF, you need to have Julia installed ([link](https://julialang.org/downloads/)). Then you can run the following commands to load the package GeCo, which we used as the underneath counterfactual explanation system.

```Julia
using Pkg; Pkg.activate("./GeCo.jl")
using GeCo
```

We provide scripts to load the data and model. For instance:
```Julia
include("scripts/credit/credit_setup_MACE.jl");
include("scripts/adult/adult_setup_MACE.jl");
include("../scripts/fico/fico-setup.jl")
```

Then run the following command to compute the rule-based explanations:
```Julia
explanations, = generate_rules(orig_instance, X, classifier, plaf)
```
for GeneticRule;
```Julia
explanations, = generate_rules(orig_instance, X, classifier, plaf, geco_initial = true, geco_mutation = true)
```
for RuleCF;
```Julia
explanation, = greedyCF(orig_instance, X, classifier, plaf)
```
for RuleCFGreedy,
where `orig_entity` is the entity to be explained, `X` is the dataset, `classifier` is the model that is explained, and plaf is the PLAF constrains.

To print out the top rule:
```Julia
explanation = explanations[1]
print_rule(orig_instance, explanation)
```
