package scaler

import (
	"math"
	"testing"
)

const tol64 = 1e-10

func TestScale(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		val  float64
		min  float64
		max  float64
		want float64
	}{
		{"midpoint", 5, 0, 10, 0.0},
		{"at_min", 0, 0, 10, -1.0},
		{"at_max", 10, 0, 10, 1.0},
		{"zero_range", 7, 5, 5, 0.0},
		{"below_min", -2, 0, 10, -1.4},
		{"above_max", 12, 0, 10, 1.4},
		{"negative_range", -6, -10, -2, 0.0},
		{"large_values", 1e12, 0, 2e12, 0.0},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			var features [36]float64
			var mins, maxs [36]float64
			features[0] = tt.val
			mins[0] = tt.min
			maxs[0] = tt.max
			Scale(&features, mins, maxs)
			if math.Abs(features[0]-tt.want) > tol64 {
				t.Errorf("Scale(%f, [%f,%f]) = %f, want %f",
					tt.val, tt.min, tt.max, features[0], tt.want)
			}
		})
	}
}

func TestScale_All36Elements(t *testing.T) {
	t.Parallel()
	var features [36]float64
	var mins, maxs [36]float64
	for i := 0; i < 36; i++ {
		features[i] = float64(i)
		mins[i] = 0
		maxs[i] = float64(2 * i)
	}
	// Element 0: val=0, range=[0,0] → 0 (zero range)
	// Element i>0: val=i, range=[0, 2i] → midpoint → 0.0
	Scale(&features, mins, maxs)

	for i := 0; i < 36; i++ {
		if math.Abs(features[i]) > tol64 {
			t.Errorf("features[%d] = %f, want 0.0", i, features[i])
		}
	}
}

func TestScale_InPlace(t *testing.T) {
	t.Parallel()
	var features [36]float64
	var mins, maxs [36]float64
	features[0] = 5
	mins[0] = 0
	maxs[0] = 10
	Scale(&features, mins, maxs)
	if features[0] != 0.0 {
		t.Errorf("expected in-place modification, got %f", features[0])
	}
}
