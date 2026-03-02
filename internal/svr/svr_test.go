package svr

import (
	"math"
	"testing"
)

const tol64 = 1e-10

func TestPredict(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		model *Model
		feat  [36]float64
		want  float64
	}{
		{
			name: "exact_match_single_sv",
			model: &Model{
				SupportVectors: make([]float64, 36), // all zeros
				Alpha:          []float64{2.5},
				Rho:            1.0,
				Gamma:          1.0,
				NSV:            1,
			},
			feat: [36]float64{}, // all zeros → dist²=0 → K=1
			want: 2.5 - 1.0,     // alpha*1 - rho
		},
		{
			name: "far_away",
			model: &Model{
				SupportVectors: make([]float64, 36),
				Alpha:          []float64{2.5},
				Rho:            3.0,
				Gamma:          1.0,
				NSV:            1,
			},
			feat: func() [36]float64 {
				var f [36]float64
				for i := range f {
					f[i] = 100 // very far
				}
				return f
			}(),
			want: -3.0, // K ≈ 0
		},
		{
			name: "two_opposite_alphas",
			model: &Model{
				SupportVectors: make([]float64, 72), // two SVs, both at origin
				Alpha:          []float64{5.0, -5.0},
				Rho:            0,
				Gamma:          1.0,
				NSV:            2,
			},
			feat: [36]float64{}, // dist²=0 for both → K=1 each
			want: 0,             // 5*1 + (-5)*1 - 0
		},
		{
			name: "gamma_zero",
			model: &Model{
				SupportVectors: make([]float64, 72),
				Alpha:          []float64{3.0, 4.0},
				Rho:            2.0,
				Gamma:          0, // exp(0)=1 always
				NSV:            2,
			},
			feat: func() [36]float64 {
				var f [36]float64
				f[0] = 999 // doesn't matter
				return f
			}(),
			want: 3.0 + 4.0 - 2.0, // sum(alpha) - rho
		},
		{
			name: "all_alphas_zero",
			model: &Model{
				SupportVectors: make([]float64, 36),
				Alpha:          []float64{0},
				Rho:            7.0,
				Gamma:          1.0,
				NSV:            1,
			},
			feat: [36]float64{},
			want: -7.0,
		},
		{
			name: "rho_zero",
			model: &Model{
				SupportVectors: make([]float64, 36),
				Alpha:          []float64{3.0},
				Rho:            0,
				Gamma:          1.0,
				NSV:            1,
			},
			feat: [36]float64{},
			want: 3.0,
		},
		{
			name: "nsv_zero",
			model: &Model{
				SupportVectors: nil,
				Alpha:          nil,
				Rho:            5.5,
				Gamma:          1.0,
				NSV:            0,
			},
			feat: [36]float64{},
			want: -5.5,
		},
		{
			name: "hand_computed",
			// gamma=1, 1 SV at origin, features=[1,0,...,0]
			// dist² = 1, K = exp(-1), score = 2.5*exp(-1) - 3.0
			model: &Model{
				SupportVectors: make([]float64, 36),
				Alpha:          []float64{2.5},
				Rho:            3.0,
				Gamma:          1.0,
				NSV:            1,
			},
			feat: func() [36]float64 {
				var f [36]float64
				f[0] = 1.0
				return f
			}(),
			want: 2.5*math.Exp(-1) - 3.0,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := Predict(tt.feat, tt.model)
			if math.Abs(got-tt.want) > 1e-8 {
				t.Errorf("Predict() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestPredict_Symmetry(t *testing.T) {
	t.Parallel()
	// RBF kernel is symmetric: K(A,B) == K(B,A)
	// So Predict(A, model_with_sv=B) should equal Predict(B, model_with_sv=A)
	var featA, featB [36]float64
	featA[0] = 1.0
	featA[1] = 2.0
	featB[0] = 3.0
	featB[1] = 4.0

	modelA := &Model{
		SupportVectors: featB[:],
		Alpha:          []float64{1.0},
		Rho:            0,
		Gamma:          0.5,
		NSV:            1,
	}
	modelB := &Model{
		SupportVectors: featA[:],
		Alpha:          []float64{1.0},
		Rho:            0,
		Gamma:          0.5,
		NSV:            1,
	}
	scoreAB := Predict(featA, modelA)
	scoreBA := Predict(featB, modelB)
	if math.Abs(scoreAB-scoreBA) > tol64 {
		t.Errorf("Symmetry violated: Predict(A,sv=B)=%f, Predict(B,sv=A)=%f", scoreAB, scoreBA)
	}
}
