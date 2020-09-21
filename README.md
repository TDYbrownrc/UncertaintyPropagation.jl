# UncertaintyPropagation.jl

UncertaintyPropagation.jl is a Julia module for propagating uncertainties through arbitrary functions. 

Currently, only functions of the form 
```julia
 function f(x)
        ...
        return y
    end
```
are supported (single input variable). However, this input variable can be array valued, so any number of inputs can be concatenated together to form the input array.



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

(result=result, uncertainty=uncertainty) = propagate(foo, x, ux)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Copyright 2020 Teledyne Scientific
Subject to MIT License (see LICENSE.md for details)