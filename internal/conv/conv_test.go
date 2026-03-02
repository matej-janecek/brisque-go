package conv

import (
	"image"
	"math"
	"testing"

	"github.com/matej/brisque-go/internal/imageutil"
)

func TestMakeGaussianKernel(t *testing.T) {
	kernel := MakeGaussianKernel(1.0, 5)
	if len(kernel) != 5 {
		t.Fatalf("expected kernel size 5, got %d", len(kernel))
	}

	// Check symmetry
	for i := 0; i < len(kernel)/2; i++ {
		if math.Abs(kernel[i]-kernel[len(kernel)-1-i]) > 1e-15 {
			t.Errorf("kernel not symmetric at index %d: %f != %f", i, kernel[i], kernel[len(kernel)-1-i])
		}
	}

	// Check normalization
	sum := 0.0
	for _, v := range kernel {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("kernel sum = %f, expected 1.0", sum)
	}

	// Center should be the maximum
	center := len(kernel) / 2
	for i, v := range kernel {
		if i != center && v > kernel[center] {
			t.Errorf("center (%f) is not maximum, index %d has %f", kernel[center], i, v)
		}
	}
}

func TestConvolve_Impulse(t *testing.T) {
	// Convolving an impulse with a kernel should give the kernel
	kernel := MakeGaussianKernel(1.0, 3)

	// Create a 5x5 image with impulse at center
	src := imageutil.NewFloatImage(image.Rect(0, 0, 5, 5))
	src.Set(2, 2, 1.0) // impulse at center

	dst := imageutil.NewFloatImage(image.Rect(0, 0, 5, 5))
	tmp := make([]float32, 5*5)

	Convolve(dst, src, kernel, tmp)

	// Output should be 3x3 (valid convolution of 5x5 with 3-wide kernel)
	if dst.Width() != 3 || dst.Height() != 3 {
		t.Fatalf("expected 3x3 output, got %dx%d", dst.Width(), dst.Height())
	}

	// The center pixel of output should be kernel[1]*kernel[1] (center*center)
	center := float64(dst.At(2, 2)) // (2,2) in absolute coordinates
	expected := kernel[1] * kernel[1]
	if math.Abs(center-expected) > 1e-6 {
		t.Errorf("center pixel = %f, expected %f", center, expected)
	}
}

func TestConvolve_Constant(t *testing.T) {
	// Convolving a constant image should give a constant output
	kernel := MakeGaussianKernel(1.0, 3)
	var constVal float32 = 42.0

	src := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	for i := range src.Pix {
		src.Pix[i] = constVal
	}

	dst := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	tmp := make([]float32, 10*10)

	Convolve(dst, src, kernel, tmp)

	// All output pixels should be constVal
	for i, v := range dst.Pix[:dst.Width()*dst.Height()] {
		if math.Abs(float64(v)-float64(constVal)) > 1e-4 {
			t.Errorf("pixel %d = %f, expected %f", i, v, constVal)
			break
		}
	}
}
