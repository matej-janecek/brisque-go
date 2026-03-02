package stats

import (
	"errors"
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

func TestFitGGD_Table(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		data       []float64
		wantAlpha  float64
		alphaTol   float64
		wantSigma2 float64
		sigmaTol   float64
		wantErr    bool
	}{
		{
			name:       "single_element",
			data:       []float64{5.0},
			wantAlpha:  0.2,
			alphaTol:   0.01,
			wantSigma2: 25.0,
			sigmaTol:   0.01,
		},
		{
			name:       "constant_nonzero",
			data:       []float64{3, 3, 3, 3},
			wantAlpha:  0.2,
			alphaTol:   0.01,
			wantSigma2: 9.0,
			sigmaTol:   0.01,
		},
		{
			name: "uniform_high_alpha",
			data: func() []float64 {
				rng := rand.New(rand.NewSource(42))
				d := make([]float64, 10000)
				for i := range d {
					d[i] = rng.Float64()*2 - 1 // [-1, 1]
				}
				return d
			}(),
			wantAlpha: 10.0, // uniform → GGD alpha → ∞, capped at 10
			alphaTol:  0.5,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			alpha, sigma2, err := FitGGD(tt.data)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if math.Abs(alpha-tt.wantAlpha) > tt.alphaTol {
				t.Errorf("alpha = %f, want %f ± %f", alpha, tt.wantAlpha, tt.alphaTol)
			}
			if tt.sigmaTol > 0 && math.Abs(sigma2-tt.wantSigma2) > tt.sigmaTol {
				t.Errorf("sigma2 = %f, want %f ± %f", sigma2, tt.wantSigma2, tt.sigmaTol)
			}
		})
	}
}

func TestFitGGD_AlphaSigmaRanges(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 10000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	alpha, sigma2, err := FitGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if alpha < 0.2 || alpha > 10 {
		t.Errorf("alpha = %f, want in [0.2, 10]", alpha)
	}
	if sigma2 < 0 {
		t.Errorf("sigma2 = %f, want >= 0", sigma2)
	}
}

func TestFitGGD_LargeData(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 100000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	// Must not panic, should complete in reasonable time
	_, _, err := FitGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFitAGGD_Table(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		data    []float64
		wantLS2 float64
		wantRS2 float64
		ls2Tol  float64
		rs2Tol  float64
	}{
		{
			name:    "single_positive",
			data:    []float64{5.0},
			wantLS2: 0,
			wantRS2: 25.0,
			ls2Tol:  0.01,
			rs2Tol:  0.01,
		},
		{
			name:    "single_negative",
			data:    []float64{-3.0},
			wantLS2: 9.0,
			wantRS2: 0,
			ls2Tol:  0.01,
			rs2Tol:  0.01,
		},
		{
			name: "symmetric",
			data: []float64{-1, -2, 1, 2},
			// leftSumSq = 1+4=5, leftCount=2 → ls2=2.5
			// rightSumSq = 1+4=5, rightCount=2 → rs2=2.5
			wantLS2: 2.5,
			wantRS2: 2.5,
			ls2Tol:  0.01,
			rs2Tol:  0.01,
		},
		{
			name: "zeros_and_positives",
			data: []float64{0, 0, 1, 2},
			// Zeros excluded from both. left=0, right: 1+4=5, count=2 → rs2=2.5
			wantLS2: 0,
			wantRS2: 2.5,
			ls2Tol:  0.01,
			rs2Tol:  0.01,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, ls2, rs2, _, err := FitAGGD(tt.data)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if math.Abs(ls2-tt.wantLS2) > tt.ls2Tol {
				t.Errorf("leftSigma2 = %f, want %f", ls2, tt.wantLS2)
			}
			if math.Abs(rs2-tt.wantRS2) > tt.rs2Tol {
				t.Errorf("rightSigma2 = %f, want %f", rs2, tt.wantRS2)
			}
		})
	}
}

func TestFitAGGD_MeanSign(t *testing.T) {
	t.Parallel()
	// rightSigma > leftSigma → mean > 0
	data := make([]float64, 10000)
	rng := rand.New(rand.NewSource(42))
	for i := range data {
		v := rng.NormFloat64()
		if v < 0 {
			data[i] = v * 0.5 // small left
		} else {
			data[i] = v * 2.0 // large right
		}
	}
	_, _, _, mean, err := FitAGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mean <= 0 {
		t.Errorf("expected mean > 0 when rightSigma > leftSigma, got %f", mean)
	}

	// Reverse: leftSigma > rightSigma → mean < 0
	for i := range data {
		v := rng.NormFloat64()
		if v < 0 {
			data[i] = v * 2.0 // large left
		} else {
			data[i] = v * 0.5 // small right
		}
	}
	_, _, _, mean2, err := FitAGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mean2 >= 0 {
		t.Errorf("expected mean < 0 when leftSigma > rightSigma, got %f", mean2)
	}
}

func TestFitAGGD_AlphaRange(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(42))
	data := make([]float64, 10000)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	alpha, ls2, rs2, _, err := FitAGGD(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if alpha < 0.2 || alpha > 10 {
		t.Errorf("alpha = %f, want in [0.2, 10]", alpha)
	}
	if ls2 < 0 {
		t.Errorf("leftSigma2 = %f, want >= 0", ls2)
	}
	if rs2 < 0 {
		t.Errorf("rightSigma2 = %f, want >= 0", rs2)
	}
}

func TestDegenerateError(t *testing.T) {
	t.Parallel()
	var err error = &DegenerateError{}
	if err.Error() == "" {
		t.Error("Error() returned empty string")
	}
	var de *DegenerateError
	if !errors.As(err, &de) {
		t.Error("errors.As failed for DegenerateError")
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
