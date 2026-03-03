package conv

import (
	"context"
	"image"
	"math"
	"math/rand"
	"testing"

	"github.com/matej-janecek/brisque-go/internal/imageutil"
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

func TestMakeGaussianKernel_Table(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name      string
		sigma     float64
		size      int
		wantLen   int
		checkFunc func(t *testing.T, k []float64)
	}{
		{
			name: "even_becomes_odd", sigma: 1.0, size: 4, wantLen: 5,
			checkFunc: nil,
		},
		{
			name: "size_1", sigma: 1.0, size: 1, wantLen: 1,
			checkFunc: func(t *testing.T, k []float64) {
				if math.Abs(k[0]-1.0) > 1e-15 {
					t.Errorf("size=1 kernel should be [1.0], got %f", k[0])
				}
			},
		},
		{
			name: "large_sigma_near_uniform", sigma: 100.0, size: 5, wantLen: 5,
			checkFunc: func(t *testing.T, k []float64) {
				// With very large sigma, all weights should be nearly equal
				expected := 1.0 / float64(len(k))
				for i, v := range k {
					if math.Abs(v-expected) > 0.01 {
						t.Errorf("k[%d] = %f, expected ~%f for large sigma", i, v, expected)
					}
				}
			},
		},
		{
			name: "small_sigma_peaked", sigma: 0.1, size: 5, wantLen: 5,
			checkFunc: func(t *testing.T, k []float64) {
				// Center should dominate
				center := len(k) / 2
				if k[center] < 0.99 {
					t.Errorf("small sigma: center = %f, expected > 0.99", k[center])
				}
			},
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			k := MakeGaussianKernel(tt.sigma, tt.size)
			if len(k) != tt.wantLen {
				t.Fatalf("len = %d, want %d", len(k), tt.wantLen)
			}
			// Check symmetry
			for i := 0; i < len(k)/2; i++ {
				if math.Abs(k[i]-k[len(k)-1-i]) > 1e-15 {
					t.Errorf("not symmetric: k[%d]=%f, k[%d]=%f", i, k[i], len(k)-1-i, k[len(k)-1-i])
				}
			}
			// Check sum
			sum := 0.0
			for _, v := range k {
				sum += v
			}
			if math.Abs(sum-1.0) > 1e-10 {
				t.Errorf("sum = %f, want 1.0", sum)
			}
			if tt.checkFunc != nil {
				tt.checkFunc(t, k)
			}
		})
	}
}

func TestConvolve_OutputDimensions(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	kernel := MakeGaussianKernel(1.0, 3)
	tmp := make([]float32, 10*10)
	Convolve(dst, src, kernel, tmp)
	if dst.Width() != 8 || dst.Height() != 8 {
		t.Errorf("10x10 k=3: output %dx%d, want 8x8", dst.Width(), dst.Height())
	}

	kernel5 := MakeGaussianKernel(1.0, 5)
	tmp5 := make([]float32, 10*10)
	Convolve(dst, src, kernel5, tmp5)
	if dst.Width() != 6 || dst.Height() != 6 {
		t.Errorf("10x10 k=5: output %dx%d, want 6x6", dst.Width(), dst.Height())
	}
}

func TestConvolve_OutputRectOrigin(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	kernel := MakeGaussianKernel(1.0, 3)
	tmp := make([]float32, 10*10)
	Convolve(dst, src, kernel, tmp)
	want := image.Rect(1, 1, 9, 9)
	if dst.Rect != want {
		t.Errorf("output Rect = %v, want %v", dst.Rect, want)
	}
}

func TestConvolve_IdentityKernel(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 5, 5))
	for i := range src.Pix {
		src.Pix[i] = float32(i)
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 5, 5))
	kernel := []float64{1.0}
	tmp := make([]float32, 5*5)
	Convolve(dst, src, kernel, tmp)

	if dst.Width() != 5 || dst.Height() != 5 {
		t.Fatalf("identity: output %dx%d, want 5x5", dst.Width(), dst.Height())
	}
	for i, v := range dst.Pix {
		if math.Abs(float64(v)-float64(src.Pix[i])) > 1e-6 {
			t.Errorf("Pix[%d]: got %f, want %f", i, v, src.Pix[i])
			break
		}
	}
}

func TestConvolve_Linearity(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(42))
	w, h := 20, 20
	kernel := MakeGaussianKernel(1.0, 3)

	img1 := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	img2 := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for i := range img1.Pix {
		img1.Pix[i] = float32(rng.Float64() * 100)
		img2.Pix[i] = float32(rng.Float64() * 100)
	}

	a, b := float32(2.5), float32(0.7)

	// conv(a*I1 + b*I2)
	combined := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for i := range combined.Pix {
		combined.Pix[i] = a*img1.Pix[i] + b*img2.Pix[i]
	}
	dstC := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	tmpC := make([]float32, w*h)
	Convolve(dstC, combined, kernel, tmpC)

	// a*conv(I1) + b*conv(I2)
	dst1 := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	tmp1 := make([]float32, w*h)
	Convolve(dst1, img1, kernel, tmp1)

	dst2 := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	tmp2 := make([]float32, w*h)
	Convolve(dst2, img2, kernel, tmp2)

	for i := range dstC.Pix {
		expected := a*dst1.Pix[i] + b*dst2.Pix[i]
		if math.Abs(float64(dstC.Pix[i])-float64(expected)) > 1e-3 {
			t.Errorf("linearity at %d: conv(combined)=%f, a*conv1+b*conv2=%f",
				i, dstC.Pix[i], expected)
			break
		}
	}
}

func TestConvolve_KernelTooLarge(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 5, 5))
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 5, 5))
	kernel := MakeGaussianKernel(1.0, 7) // k=7 on 5×5 → negative output
	tmp := make([]float32, 5*5)

	// Must not panic
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked on kernel too large: %v", r)
		}
	}()
	Convolve(dst, src, kernel, tmp)
	if dst.Width() != 0 || dst.Height() != 0 {
		t.Errorf("expected 0x0 output, got %dx%d", dst.Width(), dst.Height())
	}
}

func TestConvolveReplicate_ConstantImage(t *testing.T) {
	t.Parallel()
	w, h := 10, 10
	src := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for i := range src.Pix {
		src.Pix[i] = 42
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	kernel := MakeGaussianKernel(1.0, 3)
	tmp := make([]float32, w*h)

	err := ConvolveReplicate(context.Background(), dst, src, kernel, tmp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if dst.Width() != w || dst.Height() != h {
		t.Fatalf("output %dx%d, want %dx%d", dst.Width(), dst.Height(), w, h)
	}
	for i, v := range dst.Pix {
		if math.Abs(float64(v)-42) > 1e-4 {
			t.Errorf("Pix[%d] = %f, want 42", i, v)
			break
		}
	}
}

func TestConvolveReplicate_ContextCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	src := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	kernel := MakeGaussianKernel(1.0, 3)
	tmp := make([]float32, 100)

	err := ConvolveReplicate(ctx, dst, src, kernel, tmp)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestConvolveReplicate_MinimumSize(t *testing.T) {
	t.Parallel()
	// 7×7 image with k=7 → all border pixels
	src := imageutil.NewFloatImage(image.Rect(0, 0, 7, 7))
	for i := range src.Pix {
		src.Pix[i] = float32(i)
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	kernel := MakeGaussianKernel(1.0, 7)
	tmp := make([]float32, 49)

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("panicked on minimum size: %v", r)
		}
	}()
	err := ConvolveReplicate(context.Background(), dst, src, kernel, tmp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestConvolveReplicate_InteriorMatchesConvolve(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(42))
	w, h := 20, 20
	src := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for i := range src.Pix {
		src.Pix[i] = float32(rng.Float64() * 255)
	}

	kernel := MakeGaussianKernel(1.0, 5) // half=2
	half := len(kernel) / 2

	dstValid := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	tmpV := make([]float32, w*h)
	Convolve(dstValid, src, kernel, tmpV)

	dstRepl := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	tmpR := make([]float32, w*h)
	err := ConvolveReplicate(context.Background(), dstRepl, src, kernel, tmpR)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Interior of replicate should match valid-mode
	outW := w - len(kernel) + 1
	outH := h - len(kernel) + 1
	for y := 0; y < outH; y++ {
		for x := 0; x < outW; x++ {
			vValid := dstValid.At(x+half, y+half)
			vRepl := dstRepl.At(x+half, y+half)
			if math.Abs(float64(vValid)-float64(vRepl)) > 1e-4 {
				t.Errorf("interior (%d,%d): valid=%f, replicate=%f", x+half, y+half, vValid, vRepl)
				return
			}
		}
	}
}
