using Test
using ADUncertaintyProp

function f(x)
    sqrt(x)
end
x = 8
∂x = 1/2*x^(-1/2)
σx = 0.1
σy = sqrt(∂x^2 * σx^2)
@test propagate(f,x,σx) ≈ σy

function g(x)
    return vcat(2 .*x,-2 .*x)
end
x = 2
σx = 0.1
@test propagate(g,x,σx) ≈ [0.2,0.2]

x = [2,3,4]
σx = [0.2,0.3,0.4]
@test propagate(g,x,σx) ≈ [0.4,0.6,0.8,0.4,0.6,0.8]

function h(x)
    return sum(x)
end
x = [2,3,4]
σx = [0.1,0.2,0.3]
@test propagate(h,x,σx) ≈ sqrt(sum(σx.^2))

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

bbox1 = [10,10,50,50]
bbox2 = [9,11,47,54]
σ_bbox1 = [1,3,2.5,5]
σ_bbox2 = [0,0,0,0]

@show propagate(calc_iou,vcat(bbox1,bbox2),vcat(σ_bbox1,σ_bbox2))


