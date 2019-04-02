export QuadTreeFiltering

module QuadTreeFiltering

using ..StateEstimation
using Parameters
using LightGraphs, RegionTrees, StaticArrays

export
    CellData,
    QuadTreeFilter,
    num_cells,

    child_data,
    split_filter_node

"""
    CellData

    data associated with a hyper-rectangle cell of a QuadTreeFilter
"""
@with_kw mutable struct CellData
    id::Int       = -1
    prob::Float64 = 0.0
    area::Float64 = 0.0
end

"""
    QuadTreeFilter

    An filter that maintains a discrete distribution over an adaptively
    discretized 2D space.
"""
@with_kw mutable struct QuadTreeFilter
    root::Cell                 = Cell(SVector(0.0,0.0),SVector(0.0,0.0),CellData())
    cells::Vector{Cell}        = Vector{Cell}()
    active_cells::Vector{Int}  = Vector{Int}([1])
end
num_cells(m::QuadTreeFilter) = length(m.cells)
get_active_cells(m::QuadTreeFilter) = cells[active_cells]

"""
    child_data(m::QuadTreeFilter,data::CellData)

    A helper function to appropriate assign data to children of newly split
    nodes in the QuadTreeFilter.
"""
function child_data(m::QuadTreeFilter,data::CellData)
    [CellData(num_cells(m) + i, data.prob/4.0, data.area/4.0) for i in 1:4]
end

function split_filter_node!(m::QuadTreeFilter, cell::Cell)
    if isleaf(cell)
        split!(cell,child_data(m,cell.data))
    end
    for (c,d) in zip(children(cell),child_data(m,cell.data))
        c.data = d
        push!(m.cells,c)
        push!(m.active_cells, c.data.id)
    end
end

end
