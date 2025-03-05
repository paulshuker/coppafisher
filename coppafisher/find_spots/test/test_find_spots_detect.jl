module TestFindSpotsDetect

include("../detect.jl")

using Test
using NPZ

using .Detect


function _io_result_is_equal(
    temp_dir::String,
    image::Array{Float32, 3},
    other_inputs::Tuple,
    maxima_yxz_expected::Matrix{Int16},
    maxima_intensity_expected::Vector{Float32},
)::Bool
    image_filepath = joinpath(temp_dir, "image.npy")
    rm(image_filepath, force=true)
    npzwrite(image_filepath, image)

    args::Vector{String} = [image_filepath]
    args = vcat(args, [string(arg) for arg in other_inputs])
    Detect.detect_spots_io(args)

    maxima_yxz_filepath = joinpath(temp_dir, "maxima_yxz.npy")
    maxima_intensity_filepath = joinpath(temp_dir, "maxima_intensity.npy")
    if !isfile(maxima_yxz_filepath)
        error("Failed to find result file maxima_intensity.npy")
    end
    if !isfile(maxima_intensity_filepath)
        error("Failed to find result file maxima_yxz.npy")
    end

    maxima_yxz = npzread(maxima_yxz_filepath)
    maxima_intensity = npzread(maxima_intensity_filepath)

    if !(maxima_yxz ≈ maxima_yxz_expected)
        return false
    end
    if !(maxima_intensity ≈ maxima_intensity_expected)
        return false
    end

    return true
end

@testset "test_detect_spots" begin
    temp_dir = mktempdir()

    image_shape = (3, 4, 5)
    image = zeros(Float32, image_shape...)
    image[1, 1, 1] = 1
    intensity_thresh = 0
    radius_xy = 1
    radius_z = 1.0
    result = Detect.detect_spots(image, intensity_thresh, radius_xy, radius_z)
    maxima_yxz = result[1]
    maxima_intensity = result[2]
    @test isa(maxima_yxz, Matrix{Int16})
    @test size(maxima_yxz) == (1, 3)
    @test maxima_yxz[1] == 0
    @test isa(maxima_intensity, Vector{Float32})
    @test _io_result_is_equal(temp_dir, image, (intensity_thresh, radius_xy, radius_z), maxima_yxz, maxima_intensity)

    # Image with one isolated maxima and two nearby maxima.
    image[1, 4, 3] = 2
    image[1, 4, 5] = 2
    result = Detect.detect_spots(image, intensity_thresh, radius_xy, radius_z)
    maxima_yxz = result[1]
    maxima_intensity = result[2]
    @test size(maxima_yxz) == (3, 3)
    @test all(maxima_yxz[1, :] .== 0)
    @test all(maxima_yxz[2, :] == [0, 3, 2])
    @test all(maxima_yxz[3, :] == [0, 3, 4])
    @test maxima_intensity[1] == 1
    @test maxima_intensity[2] == 2
    @test maxima_intensity[3] == 2
    @test _io_result_is_equal(temp_dir, image, (intensity_thresh, radius_xy, radius_z), maxima_yxz, maxima_intensity)

    radius_xy = 2
    result = Detect.detect_spots(image, intensity_thresh, radius_xy, radius_z)
    maxima_yxz = result[1]
    maxima_intensity = result[2]
    @test size(maxima_yxz) == (3, 3)
    @test all(maxima_yxz[1, :] .== 0)
    @test all(maxima_yxz[2, :] == [0, 3, 2])
    @test all(maxima_yxz[3, :] == [0, 3, 4])
    @test maxima_intensity[1] == 1
    @test maxima_intensity[2] == 2
    @test maxima_intensity[3] == 2
    @test _io_result_is_equal(temp_dir, image, (intensity_thresh, radius_xy, radius_z), maxima_yxz, maxima_intensity)

    radius_xy = 1
    radius_z = 2
    result = Detect.detect_spots(image, intensity_thresh, radius_xy, radius_z)
    maxima_yxz = result[1]
    maxima_intensity = result[2]
    @test size(maxima_yxz) == (2, 3)
    @test all(maxima_yxz[1, :] .== 0)
    @test all(maxima_yxz[2, :] == [0, 3, 2]) || all(maxima_yxz[2, :] == [0, 3, 4])
    @test maxima_intensity[1] == 1
    @test maxima_intensity[2] == 2
    @test _io_result_is_equal(temp_dir, image, (intensity_thresh, radius_xy, radius_z), maxima_yxz, maxima_intensity)

    image[1, 4, 5] = 5
    result = Detect.detect_spots(image, intensity_thresh, radius_xy, radius_z)
    maxima_yxz = result[1]
    maxima_intensity = result[2]
    @test size(maxima_yxz) == (2, 3)
    @test all(maxima_yxz[1, :] .== 0)
    @test all(maxima_yxz[2, :] == [0, 3, 4])
    @test maxima_intensity[1] == 1
    @test maxima_intensity[2] == 5
    @test _io_result_is_equal(temp_dir, image, (intensity_thresh, radius_xy, radius_z), maxima_yxz, maxima_intensity)

    rm(temp_dir, recursive=true, force=true)
end

end

using .TestFindSpotsDetect
