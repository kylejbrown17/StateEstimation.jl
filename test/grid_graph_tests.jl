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
    for v1 in vertices(G)
        for v2 in vertices(G)
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
G = generate_grid_graph(4,4)
pts = [[get_prop(G,v,:x),get_prop(G,v,:y)] for v in vertices(G)]
μ = zeros(Int,nv(G))
μ[1] = 1
kdtree = KDTree()
A = adjacency_matrix(G) + SparseMatrixCSC{Int}(I,nv(G),nv(G))
x = zeros(Int,nv(G))
x[1] = 1
while sum(x) < nv(G)
    @show x = Vector{Int}(A*x .>= 1)
end

BinaryDiscreteFilter()
