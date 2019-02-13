using LightGraphs, MetaGraphs
using LinearAlgebra, StaticArrays, SparseArrays
using NearestNeighbors
using StateEstimation
using Test

function generate_grid_graph(dx,dy)
    """
    Generates a random grid graph of dimensions `dx` x `dy`.
    """
    X = vcat([collect(1:dx) for i in 1:dy]...)
    Y = vcat([i*ones(dy) for i in 1:dx]...)
    N = dx*dy
    G = MetaGraph()
    for i in 1:length(X)
        add_vertex!(G,Dict(:x=>X[i],:y=>Y[i]))
    end
    kdtree = KDTree(collect(hcat(X,Y)'))
    for v1 in vertices(G)
        idxs, dists = knn(kdtree, [get_prop(G,v1,:x),get_prop(G,v1,:y)], 5)
        for v2 in idxs #vertices(G)
            if v1 != v2
                if abs(get_prop(G,v1,:x)-get_prop(G,v2,:x)) == 1 && get_prop(G,v1,:y) == get_prop(G,v2,:y)
                    add_edge!(G,v1,v2)
                elseif abs(get_prop(G,v1,:y)-get_prop(G,v2,:y)) == 1 && get_prop(G,v1,:x) == get_prop(G,v2,:x)
                    add_edge!(G,v1,v2)
                end
            end
        end
    end
    G
end

# G represents a 4-connected grid
G = generate_grid_graph(8,8)
pts = [VecE2(get_prop(G,v,:x),get_prop(G,v,:y)) for v in vertices(G)]
μ = zeros(Int,nv(G))
μ[1] = 1
kdtree = KDTree(Matrix(hcat(pts...)))
A = adjacency_matrix(G) + SparseMatrixCSC{Int}(I,nv(G),nv(G))
sysF = BinaryReachabilityTransitionModel(A)
sysG = LinearGaussianSensor(identity_sensor(2,2),Matrix{Float64}(I,2,2))
x = zeros(Int,nv(G))
x[1] = 1

m = BinaryDiscreteFilter(pts,μ,kdtree,4,sysF,sysG)
update!(m, [2.5,2.5])
@test sum(m.μ) == m.k 
predict!(m, rand(2))
@test 0 <= sum(m.μ) <= length(pts) 
update!(m, [2.5,2.5])
@test sum(m.μ) == m.k 
