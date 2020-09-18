using Test
using UncertaintyPropagation

import Base.isapprox

Base.isapprox(x::NamedTuple, y::NamedTuple; kws...) = isapprox(collect(x),collect(y);kws...)

#module 1
function h(x)
    y = sum(x)
    return y
end

function g(x)
    y = sqrt(sum(x .^2))
    return y
end

function b(x)
    return 2 .*x
end

function u(x)
    return vcat(2 .*x,-2 .*x)
end

#module 2
function k(x)
    y = sqrt.(x)
    return y
end

function l(x)
    return vcat(x[1]+x[2],x[3]+x[2])
end

function f(x)
    t1 = k(x)
    y = b(t1)
    return y
end

function calc_iou(bboxes)
    bbox1 = bboxes[1:4]
    bbox2 = bboxes[5:8]
    xA = max(bbox1[1],bbox2[1])
    yA = max(bbox1[2],bbox2[2])
    xB = min(bbox1[3],bbox2[3])
    yB = min(bbox1[4],bbox2[4])

    AI = max((xB - xA + 1),0)*max((yB - yA + 1),0)
    Aa = (bbox1[3] - bbox1[1]+1)*((bbox1[4]- bbox1[2]+1))
    Ab = (bbox2[3] - bbox2[1]+1)*((bbox2[4]- bbox2[2]+1))

    iou = AI / (Aa + Ab - AI)
    return iou
end

@testset "All tests" begin
    x = 8
    σx = 0.2
    @testset "Scalar tests" begin
        @test propagate(k,x,σx) ≈ (y=k(x), σy=1/2*x^(-1/2)*σx)
        @test propagate(b,x,σx) ≈ (y=b(x), σy=2*σx)
        @test propagate(u,x,σx) ≈ (y=u(x), σy=[2*σx,2*σx])
        @test propagate(f,x,σx) ≈ (y=f(x), σy=x^(-1/2)*σx)
    end

    x = [2,3,4]
    σx = [0.2,0.3,0.4]
    @testset "Vector tests" begin
        @test propagate(h,x,σx) ≈ (y=h(x),σy=sqrt(sum(σx .^2)))
        @test propagate(u,x,σx) ≈ (y=u(x),σy=vcat(2 .*σx,2 .*σx))
        @test propagate(b,x,σx) ≈ (y=b(x), σy=2 .*σx)
        @test propagate(k,x,σx) ≈ (y=k(x), σy=1/2 .*x.^(-1/2) .*σx)
        @test propagate(l,x,σx) ≈ (y=l(x), σy=sqrt.([σx[1]^2 + σx[2]^2,σx[2]^2 + σx[3]^2]))
        @test propagate(f,x,σx) ≈ (y=f(x),  σy=x .^(-1/2) .*σx)
    end

end


