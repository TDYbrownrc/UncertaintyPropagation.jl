module ADUncertaintyProp

import Zygote
using LinearAlgebra

export jacobian
export propagate

function jacobian(f, x)
    y = f(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    j = Array{T, 2}(undef, n, m)
    for i in 1:n
        j[i, :] .= Zygote.gradient(x -> f(x)[i], x)[1]
    end
    return j
end

# function jacobian(f,x::Array)
#     y,back  = Zygote.pullback(f,x)
#     k  = length(y)
#     n  = length(x)
#     J  = Matrix{eltype(y)}(undef,k,n)
#     e_i = zero(x)
#     for i = 1:k
#         e_i[i] = oneunit(eltype(x))
#         J[i,:] = back(e_i)[1]
#         e_i[i] = zero(eltype(x))
#     end
#     J
# end


# if length(y) > 1
#     J = jacobian(f,x)
#     cov = Diagonal(σx)
#     σy = J * cov * J'
# else

# ∂f = Zygote.gradient(f,x)
# σy = (sqrt(sum((reduce(vcat,∂f).*reduce(vcat,σx)).^2)))

function propagate(f,x::Number,σx::Number)
    J = jacobian(f,x)
    cov = Diagonal([σx]).^2
    σy = sqrt.(diag(J * cov * J'))
    if length(σy) == 1
        σy = σy[1]
    end
    return σy
end

function propagate(f,x::Array,σx::Array)
    J = jacobian(f,x)
    cov = Diagonal(σx).^2
    σy = sqrt.(diag(J * cov * J'))
    return σy
end


end # module
