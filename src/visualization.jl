"""
Plot a 2d surface embedded in 3d

surface_2dembed!(ax, embedding, xs, ys; color=:grey)
"""
function surface_2dembed!(args...; kwargs...)
    error("Load GLMakie before using surface_2dembed!")
end

"""
Plot scalars in 2d on 2d axes

scalar_2d!(ax, coordinates, xs, ys, f; colormap=:viridis, interpolate=false)
"""
function scalar_2d!(args...; kwargs...)
    error("Load GLMakie before using scalar_2d!")
end

"""
Plot scalars on a 2d surface embedded in 3d

scalar_2dembed!(ax, coordinates, embedding, xs, ys, f; colormap=:viridis)
"""
function scalar_2dembed!(args...; kwargs...)
    error("Load GLMakie before using scalar_2dembed!")
end

"""
Plot vectors in 2d on 2d axes

vectors_2d!(ax, coordinates, xs, ys, X; spacing=1, lengthscale=1, colormap=:viridis, normalize=false)
"""
function vectors_2d!(args...; kwargs...)
    error("Load GLMakie before using vectors_2d!")
end

"""
Plot vectors on a 2d surface embedded in 3d

vectors_2dembed!(ax, coordinates, basis, embedding, xs, ys, X; spacing=1, lengthscale=1, colormap=:viridis, normalize=false)
"""
function vectors_2dembed!(args...; kwargs...)
    error("Load GLMakie before using vectors_2dembed!")
end