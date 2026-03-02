package stats

import (
	"math"
	"math/rand"
	"testing"
)

func TestFitGGD_Gaussian(t *testing.T) {
	// Generate Gaussian data (alpha should be ~2)
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 10000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}

	alpha, sigma2, err := FitGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Alpha should be close to 2.0 for Gaussian
	if math.Abs(alpha-2.0) > 0.2 {
		t.Errorf("expected alpha ≈ 2.0, got %f", alpha)
	}

	// sigma² should be close to 1.0 (unit variance)
	if math.Abs(sigma2-1.0) > 0.1 {
		t.Errorf("expected sigma² ≈ 1.0, got %f", sigma2)
	}

	t.Logf("Gaussian: alpha=%.4f, sigma²=%.4f", alpha, sigma2)
}

func TestFitGGD_Laplacian(t *testing.T) {
	// Generate Laplacian data (alpha should be ~1)
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 10000)
	for i := range data {
		u := rng.Float64() - 0.5
		if u > 0 {
			data[i] = -math.Log(1.0-2.0*u) / math.Sqrt(2.0)
		} else {
			data[i] = math.Log(1.0+2.0*u) / math.Sqrt(2.0)
		}
	}

	alpha, _, err := FitGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if math.Abs(alpha-1.0) > 0.2 {
		t.Errorf("expected alpha ≈ 1.0 for Laplacian, got %f", alpha)
	}

	t.Logf("Laplacian: alpha=%.4f", alpha)
}

func TestFitGGD_ZeroData(t *testing.T) {
	data := make([]float64, 100)
	alpha, sigma2, err := FitGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if alpha != 2.0 || sigma2 != 0.0 {
		t.Errorf("expected (2.0, 0.0) for zero data, got (%f, %f)", alpha, sigma2)
	}
}

func TestFitGGD_Empty(t *testing.T) {
	_, _, err := FitGGD(nil)
	if err == nil {
		t.Error("expected error for empty data")
	}
}

func TestFitAGGD_SymmetricGaussian(t *testing.T) {
	// Generate symmetric Gaussian data
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 10000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}

	alpha, leftSigma2, rightSigma2, mean, err := FitAGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// For symmetric Gaussian, left and right sigmas should be similar
	if math.Abs(leftSigma2-rightSigma2) > 0.2 {
		t.Errorf("expected similar left/right sigma², got %f and %f", leftSigma2, rightSigma2)
	}

	// Mean should be near zero for symmetric data
	if math.Abs(mean) > 0.1 {
		t.Errorf("expected mean ≈ 0, got %f", mean)
	}

	t.Logf("Symmetric AGGD: alpha=%.4f, leftσ²=%.4f, rightσ²=%.4f, mean=%.4f",
		alpha, leftSigma2, rightSigma2, mean)
}

func TestFitAGGD_Asymmetric(t *testing.T) {
	// Generate asymmetric data
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 10000)
	for i := range data {
		v := rng.NormFloat64()
		if v < 0 {
			data[i] = v * 0.5 // smaller left variance
		} else {
			data[i] = v * 1.5 // larger right variance
		}
	}

	alpha, leftSigma2, rightSigma2, _, err := FitAGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Right sigma should be larger than left
	if rightSigma2 <= leftSigma2 {
		t.Errorf("expected rightσ² > leftσ², got %f <= %f", rightSigma2, leftSigma2)
	}

	t.Logf("Asymmetric AGGD: alpha=%.4f, leftσ²=%.4f, rightσ²=%.4f",
		alpha, leftSigma2, rightSigma2)
}

func TestFitAGGD_Empty(t *testing.T) {
	_, _, _, _, err := FitAGGD(nil)
	if err == nil {
		t.Error("expected error for empty data")
	}
}

func TestFitAGGD_AllZero(t *testing.T) {
	data := make([]float64, 100)
	alpha, _, _, _, err := FitAGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if alpha != 2.0 {
		t.Errorf("expected alpha=2.0 for zero data, got %f", alpha)
	}
}

func BenchmarkFitGGD(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 100000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = FitGGD(data)
	}
}
