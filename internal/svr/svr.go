package svr

import "math"

// Model holds the SVR parameters for RBF kernel prediction.
type Model struct {
	// SupportVectors is a flat N×36 matrix of support vectors.
	SupportVectors []float64
	// Alpha coefficients for each support vector.
	Alpha []float64
	// Rho is the bias term.
	Rho float64
	// Gamma is the RBF kernel parameter.
	Gamma float64
	// NSV is the number of support vectors.
	NSV int
}

// Predict computes the SVR prediction: score = sum(alpha_i * K(x, sv_i)) - rho
// where K is the RBF kernel: K(x, y) = exp(-gamma * ||x - y||²).
// This function makes zero allocations.
func Predict(features [36]float64, model *Model) float64 {
	sum := 0.0
	nFeatures := 36
	svs := model.SupportVectors
	alphas := model.Alpha
	gamma := model.Gamma

	for i := 0; i < model.NSV; i++ {
		// Compute ||features - sv_i||²
		offset := i * nFeatures
		distSq := 0.0
		sv := svs[offset : offset+nFeatures]
		for j := 0; j < nFeatures; j++ {
			d := features[j] - sv[j]
			distSq += d * d
		}
		// RBF kernel
		k := math.Exp(-gamma * distSq)
		sum += alphas[i] * k
	}

	return sum - model.Rho
}
