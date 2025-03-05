module DetectIO

include("detect.jl")

using .Detect


Detect.detect_spots_io(ARGS)

end
