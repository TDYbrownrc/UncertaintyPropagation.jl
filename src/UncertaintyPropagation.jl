# Copyright (c) 2020 Teledyne Scientific
# See LICENSE.md for licensing permissions

module UncertaintyPropagation

# Zygote is the main engine for uncertainty propagation, which requires gradients from the function
import Zygote
using LinearAlgebra

# The propagate function is the main entrypoint into this module, accepting a function and paired inputs/uncertainties
# and returning a NamedTuple of the outputs and paired propagated uncertainties
export propagate

# the jacobian function is the main workhorse of the module. 
# It creates a jacobian relating the gradients of the inputs and outputs which is use to propagate.
function jacobian(f::Function,x)
    # Zygote.pullback returns f(x) and a function, back, that allows calculation of gradients of f
    y,back = Zygote.pullback(f,x)

    # gather information about input-output types
    k = length(y)
    n = length(x)

    # instantiate jacobian matrix J
    J  = Matrix{eltype(y)}(undef,k,n)

    # create a per-input unit vector for calculating gradients w.r.t each output
    e_i = zero(y)

    # If output is scalar-valued
    if k == 1
        # unit-vector becomes unit scalar of the type corresponding to the output
        e_i = oneunit(eltype(y))

        #compute gradient of f w.r.t the scalar output
        ∂f = back(e_i)[1]

        #if no gradient return zeros
        if ∂f === nothing
            return y, fill!(J,0)
        end

        if n == 1
            #if size(inputs) == 1 return scalar gradient value
            J = ∂f
        else
            #if size(inputs) > 1 return Jacobian matrix
            J[1,:] = ∂f
        end
        return y, J
    end

    # If output is vector-valued
    for i = 1:k
        # create unit vector for each of the k outputs
        e_i[i] = oneunit(eltype(y))

        #compute gradient w.r.t. output i
        ∂f = back(e_i)[1]
        if ∂f === nothing
            # handle no gradient
            ∂f = zeros(J[i,:])
        end
        # populate Jacobian matrix
        J[i,:] .= ∂f
        e_i[i] = zero(eltype(y))
    end
    return y, J
end

# Propagation function (scalar-scalar)
function propagate(f::Function,x::Number,σx::Number)
    # compute f(x) and Jacobian
    y, J = jacobian(f,x)

    # compute covariance matrix from input uncertainties
    cov = Diagonal([σx]).^2

    #compute output uncertainties using Jacobian matrix and covariance
    σy = sqrt.(diag(J * cov * J'))
    if length(σy) == 1
        # convert length 1 array to scalar value
        σy = σy[1]
    end
    #return a named tuple
    return (result=y, uncertainty=σy)
end

# Propagation function (array-array)
# see comments in propagate(f::Function, x::Number, σx::Number) for line-by-line explanation
function propagate(f::Function,x::Array,σx::Array)
    y, J = jacobian(f,x)
    cov = Diagonal(σx).^2
    σy = sqrt.(diag(J * cov * J'))
    if length(σy) == 1
        σy = σy[1]
    end
    return (result=y, uncertainty=σy)
end

end # module

# This code was developed through partial support by the Defense Advanced Research Projects Agency (DARPA), Lifelong Learning Machines, Contract No. FA8650-18-C-7831
