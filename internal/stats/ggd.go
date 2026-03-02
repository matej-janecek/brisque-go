package stats

import "math"

// gammaFunc wraps math.Gamma for readability.
func gammaFunc(x float64) float64 { return math.Gamma(x) }

// FitGGD estimates the shape (alpha) and variance (sigma²) parameters of a
// Generalized Gaussian Distribution from the data. It uses the moment-ratio
// method: r = (mean(|x|))² / mean(x²), then solves for alpha via
// Newton-Raphson with bisection fallback.
//
// Returns alpha (shape), sigma² (variance), and any error.
func FitGGD(data []float64) (alpha, sigma2 float64, err error) {
	n := len(data)
	if n == 0 {
		return 0, 0, &DegenerateError{}
	}

	// Compute mean(|x|) and mean(x²)
	var sumAbs, sumSq float64
	for _, v := range data {
		sumAbs += math.Abs(v)
		sumSq += v * v
	}
	meanAbs := sumAbs / float64(n)
	meanSq := sumSq / float64(n)

	if meanSq < 1e-10 {
		// Near-zero variance: return Gaussian-like defaults
		return 2.0, 0.0, nil
	}

	rho := meanAbs * meanAbs / meanSq

	// rho = Gamma(2/alpha)^2 / (Gamma(1/alpha) * Gamma(3/alpha))
	// Solve for alpha using Newton-Raphson on [0.2, 10]
	alpha = solveGGDAlpha(rho)

	// sigma² = mean(x²)
	sigma2 = meanSq

	return alpha, sigma2, nil
}

// solveGGDAlpha solves Gamma(2/a)^2 / (Gamma(1/a)*Gamma(3/a)) = rho
// for a in [0.2, 10] using Newton-Raphson with bisection fallback.
func solveGGDAlpha(rho float64) float64 {
	// Clamp rho to valid range
	if rho <= 0 {
		return 10.0
	}
	if rho >= 1 {
		return 0.2
	}

	// Bisection bounds
	lo, hi := 0.2, 10.0
	const maxIter = 50
	const tol = 1e-10

	// Use bisection — reliable and sufficient for this problem
	for i := 0; i < maxIter; i++ {
		mid := (lo + hi) / 2.0
		val := gammaRatio(mid)
		if math.Abs(val-rho) < tol {
			return mid
		}
		// gammaRatio is monotonically increasing with alpha
		if val < rho {
			lo = mid // need higher alpha
		} else {
			hi = mid // need lower alpha
		}
	}
	return (lo + hi) / 2.0
}

// gammaRatio computes Gamma(2/a)^2 / (Gamma(1/a)*Gamma(3/a))
func gammaRatio(a float64) float64 {
	g1 := gammaFunc(1.0 / a)
	g2 := gammaFunc(2.0 / a)
	g3 := gammaFunc(3.0 / a)
	if g1 == 0 || g3 == 0 {
		return 0
	}
	return (g2 * g2) / (g1 * g3)
}

// DegenerateError indicates that statistical fitting failed due to
// degenerate input data (e.g., empty or near-zero variance).
type DegenerateError struct{}

func (e *DegenerateError) Error() string {
	return "degenerate distribution"
}
