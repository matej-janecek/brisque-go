package conv

import "math"

// MakeGaussianKernel returns a normalized 1D Gaussian kernel.
// size must be odd; sigma > 0.
func MakeGaussianKernel(sigma float64, size int) []float64 {
	if size%2 == 0 {
		size++
	}
	kernel := make([]float64, size)
	center := size / 2
	twoSigma2 := 2.0 * sigma * sigma
	sum := 0.0
	for i := 0; i < size; i++ {
		d := float64(i - center)
		kernel[i] = math.Exp(-(d * d) / twoSigma2)
		sum += kernel[i]
	}
	// Normalize
	for i := range kernel {
		kernel[i] /= sum
	}
	return kernel
}
