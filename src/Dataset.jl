module DatasetModule

import DynamicQuantities:
    AbstractDimensions,
    AbstractQuantity,
    Dimensions,
    SymbolicDimensions,
    Quantity,
    uparse,
    sym_uparse,
    ustrip,
    DEFAULT_DIM_BASE_TYPE

import ..UtilsModule: subscriptify
import ..ProgramConstantsModule: BATCH_DIM, FEATURE_DIM, DATA_TYPE, LOSS_TYPE
#! format: off
import ...deprecate_varmap
#! format: on

const QuantityLike = Union{AbstractDimensions,AbstractQuantity,AbstractString,Real}

"""
    Dataset{T<:DATA_TYPE,L<:LOSS_TYPE}

# Fields

- `X::AbstractMatrix{T}`: The input features, with shape `(nfeatures, n)`.
- `y::AbstractVector{T}`: The desired output values, with shape `(n,)`.
- `n::Int`: The number of samples.
- `nfeatures::Int`: The number of features.
- `weighted::Bool`: Whether the dataset is non-uniformly weighted.
- `weights::Union{AbstractVector{T},Nothing}`: If the dataset is weighted,
    these specify the per-sample weight (with shape `(n,)`).
- `extra::NamedTuple`: Extra information to pass to a custom evaluation
    function. Since this is an arbitrary named tuple, you could pass
    any sort of dataset you wish to here.
- `avg_y`: The average value of `y` (weighted, if `weights` are passed).
- `use_baseline`: Whether to use a baseline loss. This will be set to `false`
    if the baseline loss is calculated to be `Inf`.
- `baseline_loss`: The loss of a constant function which predicts the average
    value of `y`. This is loss-dependent and should be updated with
    `update_baseline_loss!`.
- `variable_names::Array{String,1}`: The names of the features,
    with shape `(nfeatures,)`.
- `pretty_variable_names::Array{String,1}`: A version of `variable_names`
    but for printing to the terminal (e.g., with unicode versions).
- `y_variable_name::String`: The name of the output variable.
- `X_units`: Unit information of `X`. When used, this is a vector
    of `DynamicQuantities.Quantity{<:Any,<:Dimensions}` with shape `(nfeatures,)`.
- `y_units`: Unit information of `y`. When used, this is a single
    `DynamicQuantities.Quantity{<:Any,<:Dimensions}`.
- `X_sym_units`: Unit information of `X`. When used, this is a vector
    of `DynamicQuantities.Quantity{<:Any,<:SymbolicDimensions}` with shape `(nfeatures,)`.
- `y_sym_units`: Unit information of `y`. When used, this is a single
    `DynamicQuantities.Quantity{<:Any,<:SymbolicDimensions}`.
"""
mutable struct Dataset{
    T<:DATA_TYPE,
    L<:LOSS_TYPE,
    AX<:AbstractMatrix{T},
    AY<:Union{AbstractVector{T},Nothing},
    AW<:Union{AbstractVector{T},Nothing},
    NT<:NamedTuple,
    XU<:Union{AbstractVector{<:Quantity},Nothing},
    YU<:Union{Quantity,Nothing},
    XUS<:Union{AbstractVector{<:Quantity},Nothing},
    YUS<:Union{Quantity,Nothing},
}
    X::AX
    y::AY
    n::Int
    nfeatures::Int
    weighted::Bool
    weights::AW
    extra::NT
    avg_y::Union{T,Nothing}
    use_baseline::Bool
    baseline_loss::L
    variable_names::Array{String,1}
    pretty_variable_names::Array{String,1}
    y_variable_name::String
    X_units::XU
    y_units::YU
    X_sym_units::XUS
    y_sym_units::YUS
end

"""
    Dataset(X::AbstractMatrix{T}, y::Union{AbstractVector{T},Nothing}=nothing;
            weights::Union{AbstractVector{T}, Nothing}=nothing,
            variable_names::Union{Array{String, 1}, Nothing}=nothing,
            y_variable_name::Union{String,Nothing}=nothing,
            X_units::Union{AbstractVector{<:QuantityLike}, Nothing}=nothing,
            y_units::Union{QuantityLike, Nothing}=nothing,
            extra::NamedTuple=NamedTuple(),
            loss_type::Type=Nothing,
    )

Construct a dataset to pass between internal functions.
"""
function Dataset(
    X::AbstractMatrix{T},
    y::Union{AbstractVector{T},Nothing}=nothing;
    weights::Union{AbstractVector{T},Nothing}=nothing,
    variable_names::Union{Array{String,1},Nothing}=nothing,
    y_variable_name::Union{String,Nothing}=nothing,
    extra::NamedTuple=NamedTuple(),
    loss_type::Type{Linit}=Nothing,
    X_units::Union{AbstractVector,Nothing}=nothing,
    y_units=nothing,
    # Deprecated:
    varMap=nothing,
) where {T<:DATA_TYPE,Linit}
    Base.require_one_based_indexing(X)
    y !== nothing && Base.require_one_based_indexing(y)
    # Deprecation warning:
    variable_names = deprecate_varmap(variable_names, varMap, :Dataset)

    n = size(X, BATCH_DIM)
    nfeatures = size(X, FEATURE_DIM)
    weighted = weights !== nothing
    (variable_names, pretty_variable_names) = if variable_names === nothing
        ["x$(i)" for i in 1:nfeatures], ["x$(subscriptify(i))" for i in 1:nfeatures]
    else
        (variable_names, variable_names)
    end
    y_variable_name = if y_variable_name === nothing
        "y" ∉ variable_names ? "y" : "target"
    else
        y_variable_name
    end
    avg_y = if y === nothing
        nothing
    else
        if weighted
            sum(y .* weights) / sum(weights)
        else
            sum(y) / n
        end
    end
    out_loss_type = (Linit === Nothing) ? T : Linit
    use_baseline = true
    baseline = one(out_loss_type)
    D = Dimensions{DEFAULT_DIM_BASE_TYPE}
    SD = SymbolicDimensions{DEFAULT_DIM_BASE_TYPE}
    X_si_units = get_units(T, D, X_units, uparse)
    y_si_units = get_units(T, D, y_units, uparse)
    X_sym_units = get_units(T, SD, X_units, sym_uparse)
    y_sym_units = get_units(T, SD, y_units, sym_uparse)

    error_on_mismatched_size(nfeatures, X_si_units)

    return Dataset{
        T,
        out_loss_type,
        typeof(X),
        typeof(y),
        typeof(weights),
        typeof(extra),
        typeof(X_si_units),
        typeof(y_si_units),
        typeof(X_sym_units),
        typeof(y_sym_units),
    }(
        X,
        y,
        n,
        nfeatures,
        weighted,
        weights,
        extra,
        avg_y,
        use_baseline,
        baseline,
        variable_names,
        pretty_variable_names,
        y_variable_name,
        X_si_units,
        y_si_units,
        X_sym_units,
        y_sym_units,
    )
end
function Dataset(
    X::AbstractMatrix,
    y::Union{<:AbstractVector,Nothing}=nothing;
    weights::Union{<:AbstractVector,Nothing}=nothing,
    kws...,
)
    T = promote_type(
        eltype(X),
        (y === nothing) ? eltype(X) : eltype(y),
        (weights === nothing) ? eltype(X) : eltype(weights),
    )
    X = Base.Fix1(convert, T).(X)
    if y !== nothing
        y = Base.Fix1(convert, T).(y)
    end
    if weights !== nothing
        weights = Base.Fix1(convert, T).(weights)
    end
    return Dataset(X, y; weights=weights, kws...)
end

# Base
function get_units(_, _, ::Nothing, ::Function)
    return nothing
end
function get_units(::Type{T}, ::Type{D}, x::AbstractString, f::Function) where {T,D}
    return convert(Quantity{T,D}, f(x))
end
function get_units(::Type{T}, ::Type{D}, x::Quantity, ::Function) where {T,D}
    return convert(Quantity{T,D}, x)
end
function get_units(::Type{T}, ::Type{D}, x::AbstractDimensions, ::Function) where {T,D}
    return convert(Quantity{T,D}, Quantity(one(T), x))
end
function get_units(::Type{T}, ::Type{D}, x::Real, ::Function) where {T,D}
    return Quantity(convert(T, x), D)::Quantity{T,D}
end

# Derived
function get_units(::Type{T}, ::Type{D}, x::AbstractVector, f::Function) where {T,D}
    return Quantity{T,D}[get_units(T, D, xi, f) for xi in x]
end

function error_on_mismatched_size(_, ::Nothing)
    return nothing
end
function error_on_mismatched_size(nfeatures, X_units::AbstractVector)
    if nfeatures != length(X_units)
        error(
            "Number of features ($(nfeatures)) does not match number of units ($(length(X_units)))",
        )
    end
    return nothing
end

function has_units(dataset::Dataset)
    return dataset.X_units !== nothing || dataset.y_units !== nothing
end

end
