package stats

import "math"

// FitAGGD estimates the parameters of an Asymmetric Generalized Gaussian
// Distribution: shape (alpha), left sigma², right sigma², and mean (eta).
// It splits the data into left (negative) and right (positive) halves
// and computes moments separately.
//
// Reference: Mittal et al., "No-Reference Image Quality Assessment in the
// Spatial Domain", IEEE TIP 2012.
func FitAGGD(data []float64) (alpha, leftSigma2, rightSigma2, mean float64, err error) {
	n := len(data)
	if n == 0 {
		return 0, 0, 0, 0, &degenerateError{}
	}

	// Split into left and right
	var leftSumSq, rightSumSq float64
	var leftCount, rightCount int
	for _, v := range data {
		if v < 0 {
			leftSumSq += v * v
			leftCount++
		} else if v > 0 {
			rightSumSq += v * v
			rightCount++
		}
	}

	// Handle degenerate cases
	if leftCount == 0 && rightCount == 0 {
		return 2.0, 0.0, 0.0, 0.0, nil
	}

	if leftCount == 0 {
		leftSigma2 = 0
	} else {
		leftSigma2 = leftSumSq / float64(leftCount)
	}
	if rightCount == 0 {
		rightSigma2 = 0
	} else {
		rightSigma2 = rightSumSq / float64(rightCount)
	}

	sigmaL := math.Sqrt(leftSigma2)
	sigmaR := math.Sqrt(rightSigma2)

	// If both sigmas are near-zero, return defaults
	if sigmaL+sigmaR < 1e-10 {
		return 2.0, 0.0, 0.0, 0.0, nil
	}

	// Compute the moment ratio: r_hat = (mean(|x|))^2 / mean(x^2)
	var sumAbs, sumSq float64
	for _, v := range data {
		sumAbs += math.Abs(v)
		sumSq += v * v
	}
	meanAbs := sumAbs / float64(n)
	meanSq := sumSq / float64(n)

	if meanSq < 1e-10 {
		return 2.0, leftSigma2, rightSigma2, 0.0, nil
	}

	rHat := (meanAbs * meanAbs) / meanSq

	// gammaHat = sigmaL / sigmaR
	gammaHat := sigmaL / sigmaR
	if sigmaR < 1e-10 {
		gammaHat = 1.0
	}

	// Normalize rhat:
	// rhat_norm = rhat * (gammahat^3 + 1) * (gammahat + 1) / (gammahat^2 + 1)^2
	gh2 := gammaHat * gammaHat
	gh3 := gh2 * gammaHat
	denom := (gh2 + 1.0) * (gh2 + 1.0)
	if denom < 1e-10 {
		denom = 1e-10
	}
	rHatNorm := rHat * (gh3 + 1.0) * (gammaHat + 1.0) / denom

	// Solve for alpha: gammaRatio(alpha) = rHatNorm
	alpha = solveAGGDAlpha(rHatNorm)

	// Compute mean of AGGD:
	// eta = (sigmaR - sigmaL) * Gamma(2/alpha) / sqrt(Gamma(1/alpha) * Gamma(3/alpha))
	g1 := gammaFunc(1.0 / alpha)
	g2 := gammaFunc(2.0 / alpha)
	g3 := gammaFunc(3.0 / alpha)
	dg := g1 * g3
	if dg < 1e-20 {
		mean = 0
	} else {
		mean = (sigmaR - sigmaL) * g2 / math.Sqrt(dg)
	}

	return alpha, leftSigma2, rightSigma2, mean, nil
}

// solveAGGDAlpha solves gammaRatio(alpha) = rHatNorm for alpha.
func solveAGGDAlpha(rHatNorm float64) float64 {
	if rHatNorm <= 0 || math.IsNaN(rHatNorm) || math.IsInf(rHatNorm, 0) {
		return 2.0
	}

	lo, hi := 0.2, 10.0
	const maxIter = 50
	const tol = 1e-10

	for i := 0; i < maxIter; i++ {
		mid := (lo + hi) / 2.0
		val := gammaRatio(mid)
		if math.Abs(val-rHatNorm) < tol {
			return mid
		}
		// gammaRatio is monotonically increasing with alpha
		if val < rHatNorm {
			lo = mid // need higher alpha
		} else {
			hi = mid // need lower alpha
		}
	}
	return (lo + hi) / 2.0
}
