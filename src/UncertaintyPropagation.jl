module UncertaintyPropagation

import Zygote
using LinearAlgebra

export run_and_propagate
export propagate

function jacobian(f::Function,x)
    y,back = Zygote.pullback(f,x)
    k = length(y)
    n = length(x)
    J  = Matrix{eltype(y)}(undef,k,n)
    e_i = zero(y)

    if k == 1
        e_i = oneunit(eltype(y))
        ∂f = back(e_i)[1]
        if ∂f === nothing
            return y, fill!(J,0)
        end
        if n == 1
            J = ∂f
        else
            J[1,:] = ∂f
        end
        return y, J
    end

    for i = 1:k
        e_i[i] = oneunit(eltype(y))
        ∂f = back(e_i)[1]
        if ∂f === nothing
            ∂f = zeros(J[i,:])
        end
        J[i,:] .= ∂f
        e_i[i] = zero(eltype(y))
    end
    return y, J
end

function propagate(f::Function,x::Number,σx::Number)
    y, J = jacobian(f,x)
    cov = Diagonal([σx]).^2
    σy = sqrt.(diag(J * cov * J'))
    if length(σy) == 1
        σy = σy[1]
    end
    return σy
end

function propagate(f::Function,x::Array,σx::Array)
    y, J = jacobian(f,x)
    cov = Diagonal(σx).^2
    σy = sqrt.(diag(J * cov * J'))
    return σy
end

function run_and_propagate(f::Function,x::Array,σx::Array)
    y, J = jacobian(f,x)
    cov = Diagonal(σx).^2
    σy = sqrt.(diag(J * cov * J'))
    if length(σy) == 1
        σy = σy[1]
    end
    return y, σy
end

function run_and_propagate(f::Function,x::Number,σx::Number)
    y, J = jacobian(f,x)
    cov = Diagonal([σx]).^2
    σy = sqrt.(diag(J * cov * J'))
    if length(σy) == 1
        σy = σy[1]
    end
    return y, σy
end

end # module
