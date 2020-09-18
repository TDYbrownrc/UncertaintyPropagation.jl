# UncertaintyPropagation.jl

UncertaintyPropagation.jl is a Julia module for propagating uncertainties through arbitrary functions.



## Installation

```julia
using Pkg
Pkg.add("git@ssh.dev.azure.com:v3/TDY-IntelligentSystems/L2M/UncertaintyPropagation.jl")
```

## Usage

```julia
using UncertaintyPropagation

function foo(x)
    #do something to x here
    return x
end

x = [1,2,3]
ux = [0.1,0.3,0.2]

run_and_propagate(foo, x, ux)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
