module Detect

using Base
using NearestNeighbors
using NPZ


function detect_spots_io(args::Vector{String})::Nothing
    if length(args) != 4
        error("detect_spots_io requires four arguments")
    end

    image_filepath = args[1]
    intensity_thresh = parse(Float32, args[2])
    radius_xy = parse(Float32, args[3])
    radius_z = parse(Float32, args[4])

    if !isfile(image_filepath)
        error("No image file found at " * image_filepath)
    end

    image = npzread(image_filepath)

    result = Detect.detect_spots(image, intensity_thresh, radius_xy, radius_z)
    maxima_yxz = result[1]
    maxima_intensity = result[2]

    maxima_yxz_filepath = joinpath(dirname(image_filepath), "maxima_yxz.npy")
    maxima_intensity_filepath = joinpath(dirname(image_filepath), "maxima_intensity.npy")

    # Overwrite any pre-existing results.
    rm(maxima_yxz_filepath, force=true)
    rm(maxima_intensity_filepath, force=true)
    npzwrite(maxima_yxz_filepath, maxima_yxz)
    npzwrite(maxima_intensity_filepath, maxima_intensity)
end


function detect_spots(
    image::Array{Float32, 3},
    intensity_thresh::Number,
    radius_xy::Number,
    radius_z::Number
)::Tuple{Matrix{Int16},Vector{Float32}}
    """
    The same function as the Python version located at `coppafisher/find_spots/detect.py`.

    The only difference is that removal of duplicate spots is always on.
    """
    if ndims(image) != 3
        error("Image must be three dimensional")
    end
    if typeof(image) != Array{Float32, 3}
        error("Image must be Array{Float32, 3}")
    end
    if intensity_thresh < 0
        error("intensity_thresh must be >= 0")
    end
    if radius_xy <= 0
        error("radius_xy must be > 0")
    end
    if radius_z <= 0
        error("radius_z must be > 0")
    end

    # Gather all image pixels above intensity_thresh.
    indices = findall(x -> x > intensity_thresh, image)
    n_spots = length(indices)
    maxima_locations = Matrix{Int16}(undef, 3, n_spots)
    maxima_intensities = Vector{Float32}(undef, n_spots)
    for (i, index) in enumerate(indices)
        maxima_locations[:, i] = [index[1], index[2], index[3]]
        maxima_intensities[i] = image[index]
    end
    indices = nothing

    # Find the nearest neighbour(s) for every maxima.
    maxima_locations_norm = copy(maxima_locations)
    maxima_locations_norm = convert(Matrix{Float32}, maxima_locations_norm)
    maxima_locations_norm[3, :] .*= Float32(radius_xy / radius_z)
    maxima_locations_norm[3, :] .+= eps(Float32)
    tree = KDTree(maxima_locations_norm)
    # The Julia code captures inclusively from the radius compared to Python, so I subtract a small amount off.
    in_range_radius = Float32(radius_xy) - eps(Float32)
    pairs::Vector{Vector{Int64}} = inrange(tree, maxima_locations_norm, in_range_radius)
    maxima_locations_norm = nothing

    # Keep the most intense maxima if there are nearby neighbours.
    keep_maxima = Bool[length(pair) == 1 for pair in pairs]
    for (i, i_pairs) in enumerate(pairs)
        if any(keep_maxima[i_pairs])
            # A near neighbour has already been kept.
            continue
        end
        if all(maxima_intensities[i] .>= maxima_intensities[i_pairs])
            keep_maxima[i_pairs] .= false
            keep_maxima[i] = true
        end
    end
    maxima_locations = maxima_locations[:, keep_maxima]
    maxima_intensities = maxima_intensities[keep_maxima]

    # (3, n_spots) -> (n_spots, 3).
    maxima_locations = permutedims(maxima_locations)

    # Maxima locations indexing starts at 1! We convert it back to starting at 0 here.
    maxima_locations = broadcast(-, maxima_locations, 1)

    return (maxima_locations, maxima_intensities)
end

end

using .Detect
