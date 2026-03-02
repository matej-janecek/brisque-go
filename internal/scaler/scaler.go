package scaler

// Scale normalizes features to the [-1, 1] range using min-max scaling.
// features is modified in place.
func Scale(features *[36]float64, mins, maxs [36]float64) {
	for i := 0; i < 36; i++ {
		rng := maxs[i] - mins[i]
		if rng == 0 {
			features[i] = 0
		} else {
			features[i] = -1.0 + 2.0*(features[i]-mins[i])/rng
		}
	}
}
