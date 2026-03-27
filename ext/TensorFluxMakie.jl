module TensorFluxMakie

using TensorFlux
using GLMakie

"""
Plot a 2d surface embedded in 3d
"""
function TensorFlux.surface_2dembed!(ax, embedding, xs, ys; color=:grey)
    us = [embedding(x, y)[1] for x in xs, y in ys]
    vs = [embedding(x, y)[2] for x in xs, y in ys]
    ws = [embedding(x, y)[3] for x in xs, y in ys]
    surface!(ax, us, vs, ws,
        colormap=[color, color]
    )
end

"""
Plot scalars in 2d on 2d axes
"""
function TensorFlux.scalar_2d!(ax, coordinates, xs, ys, f; colormap=:viridis, interpolate=false)
    u, v = coordinates
    scalars = [evaluate(f, Dict(u=>x, v=>y)) for x in xs, y in ys]
    heatmap!(ax, xs, ys, scalars, 
        colormap=colormap, interpolate=interpolate
    )
end

"""
Plot scalars on a 2d surface embedded in 3d
"""
function TensorFlux.scalar_2dembed!(ax, coordinates, embedding, xs, ys, f; colormap=:viridis)
    u, v = coordinates
    us = [embedding(x, y)[1] for x in xs, y in ys]
    vs = [embedding(x, y)[2] for x in xs, y in ys]
    ws = [embedding(x, y)[3] for x in xs, y in ys]
    scalars = [evaluate(f, Dict(u=>x, v=>y)) for x in xs, y in ys]
    surface!(ax, us, vs, ws, color=scalars,
        colormap=colormap
    )
end

"""
Plot vectors in 2d on 2d axes
"""
function TensorFlux.vectors_2d!(ax, coordinates, xs, ys, X; spacing=1, lengthscale=1, colormap=:viridis, normalize=false)
    u, v = coordinates
    grid = [(x, y) for x in xs[begin:spacing:end], y in ys[begin:spacing:end]]
    vecs = [evaluate(X, Dict(u=>x, v=>y)).data for (x, y) in grid]
    lengths = [hypot(v...) for v in vecs]
    clim = maximum(abs.(lengths))
    arrows2d!(ax, grid, vecs, color=lengths, colorrange=(-clim, clim),
        lengthscale=lengthscale, colormap=colormap, normalize=normalize
    )
end

"""
Plot vectors on a 2d surface embedded in 3d
"""
function TensorFlux.vectors_2dembed!(ax, coordinates, basis, embedding, xs, ys, X; spacing=1, lengthscale=1, colormap=:viridis, normalize=false)
    u, v = coordinates
    grid = [(x, y) for x in xs[begin:spacing:end], y in ys[begin:spacing:end]]
    grid3 = [Point3f(embedding(x, y)) for (x, y) in grid]
    vecs = [Vec3f(evaluate(X[:i] * basis[:i], Dict(u=>x, v=>y)).data) for (x, y) in grid]
    lengths = [hypot(v...) for v in vecs]
    clim = maximum(abs.(lengths))
    arrows3d!(ax, grid3, vecs, color=vec(lengths), colorrange=(-clim, clim),
        lengthscale=lengthscale, colormap=colormap, normalize=normalize
    )
end

end