# include("fico_model.jl")
using CSV, Statistics, DataFrames

path = "/Users/gzx/Desktop/research/geco/GeCo.jl/data/fico"
X = CSV.File(path*"/credit_model.csv") |> DataFrame
y = X[!, 24]
X = select!(X, Not(24))
for feature in String.(names(X))
    X[!,feature] = convert.(Float64,X[!,feature])
end
orig_instance = X[6, :]
classifier = GeCo.FicoClassifier.FICO_CLASSIFIER()
plaf = PLAFProgram()