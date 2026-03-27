# TensorFlux.jl
A differential geometry library that stays true to the mathematical notation and Einstein summation convention

## Installation
TensorFlux.jl can be installed by either running
```julia
julia> Pkg; 
julia> Pkg.add("https://github.com/contraflux/TensorFlux.jl")
```
or by pressing `]` and running
```julia
pkg> add https://github.com/contraflux/TensorFlux.jl
```
Then it can be imported with
```julia
julia> using TensorFlux
```

## Quick Start

Contracting two tensors
```julia
julia> L = Tensor([[2, 1]', [-1, 3]'])
julia> v = Tensor([1, 2])
julia> L[:i][:j] * v[:j]  # matrix-vector product
(1, 0)-Tensor:
[4, 5]
  (:contra,)
  (:i,), ()
```

Computing the Riemann curvature tensor on a 2-sphere:
```julia
julia> using Symbolics
julia> @variables u v
julia> basis = Basis([
    Tensor([1, 0]),
    Tensor([0, sin(u)])
])
julia> simplify(riemann((u, v), basis))
(1, 3)-Tensor:
Num[0.0 0.0; 0.0 -1.0;;; 0.0 sin(u)^2; 0 0;;;; 0.0 0; 1.0 0;;; -(sin(u)^2) 0; 0 0]
  (:contra, :co, :co, :co)
```

## Features
### Algebra
Tensors and bases of tensors, with algebra operations including contraction, scaling, addition, the tensor product, wedge product for differential forms, dot product for vectors, and symmetrization/antisymmetrization.
### Geometry
Tools for differential geometry, including the metric, connection coefficients, Lie bracket, Ricci scalar, and Riemann, Ricci, and Einstein tensors.
### Calculus
Tools for tensor calculus, with the partial, covariant, and exterior derivatives, and Hodge star
### Symbolic
Symbolic tensor components via Symbolics.jl, enabling exact computation of derivatives and geometric quantities.

## API
### Types
`Tensor{T, R}` - An arbitrary rank R (m, n)-tensor of a type T

`KroneckerDelta` - The Kronecker Delta δ

`LeviCivita` - The Levi-Civita Symbol ε

`PartialDerivative{N}` - The partial derivative operator ∂ on N coordinates

`CovariantDerivative` - The covariant derivative operator ∇

`ExteriorDerivative` - The exterior derivative operator d

`HodgeStar` - The Hodge Star operator

### Functions
**General**

`Tensor()` - Constructor for `Tensor{T, R}`

`getindex()` - Einstein convention indexing

**Algebra**

`⊗` - Tensor product

`*` - Tensor scaling and contraction

`+` - Tensor addition

`-` - Tensor subtraction

`⋅` - Dot product for (1, 0)-tensors

`∧` - Wedge product for (0, p)-tensors

`symmetrize()` - Symmetrize a tensor

`antisymmetrize()` - Antisymmetrize a tensor

**Geometry**

`metric()` - Metric tensor from a basis

`inv()` - Invert a (2, 0) or (0, 2)-tensor

`minkowski()` - Find the Minkowski norm on two vectors

`christoffel()` - Compute the Levi-Civita Connection coefficients

`lie()` - Compute the Lie bracket of two vectors

`riemann()` - Compute the Riemann Curvature Tensor

`ricci()` - Compute the Ricci Curvature Tensor

`ricci_scalar()` - Compute the Ricci Scalar

`einstein()` - Compute the Einstein Tensor