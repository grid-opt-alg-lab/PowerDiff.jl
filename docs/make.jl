using Documenter
using PowerModelsDiff

makedocs(
    sitename = "PowerModelsDiff.jl",
    modules = [PowerModelsDiff],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Sensitivity API" => "sensitivity-api.md",
        "Mathematical Background" => "math.md",
        "Advanced Topics" => "advanced.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/samtalki/PowerModelsDiff.jl.git",
    devbranch = "main",
)
