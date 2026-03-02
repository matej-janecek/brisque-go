package features

import (
	"image"
	"math"
	"testing"

	"github.com/matej/brisque-go/internal/imageutil"
)

const tol32 = 1e-4

func TestResizeCubicHalf_ConstantImage(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
	for i := range src.Pix {
		src.Pix[i] = 42.0
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	ResizeCubicHalf(dst, src)
	if dst.Width() != 5 || dst.Height() != 5 {
		t.Fatalf("output size %dx%d, want 5x5", dst.Width(), dst.Height())
	}
	for i, v := range dst.Pix {
		if math.Abs(float64(v)-42.0) > tol32 {
			t.Errorf("Pix[%d] = %f, want 42.0", i, v)
			break
		}
	}
}

func TestResizeCubicHalf_2x2(t *testing.T) {
	t.Parallel()
	// 2×2 → 1×1
	// Center source (sx=0, sy=0), weights [-0.09375, 0.59375, 0.59375, -0.09375]
	// With BORDER_REFLECT_101: for size=2
	//   reflect101(-1, 2) = 1
	//   reflect101(0, 2) = 0
	//   reflect101(1, 2) = 1
	//   reflect101(2, 2) = 0
	// So source positions for both x and y: [1, 0, 1, 0]
	src := imageutil.NewFloatImage(image.Rect(0, 0, 2, 2))
	src.Pix[0] = 10 // (0,0)
	src.Pix[1] = 20 // (1,0)
	src.Pix[2] = 30 // (0,1)
	src.Pix[3] = 40 // (1,1)

	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	ResizeCubicHalf(dst, src)

	if dst.Width() != 1 || dst.Height() != 1 {
		t.Fatalf("output size %dx%d, want 1x1", dst.Width(), dst.Height())
	}

	// Compute expected: for dx=0, dy=0 (sx=0, sy=0)
	// x positions: reflect101(-1,2)=1, reflect101(0,2)=0, reflect101(1,2)=1, reflect101(2,2)=0
	// y positions: same
	w := cubicWeight
	// Source values at (px, py):
	// val(px, py) uses src row py
	vals := func(px, py int) float64 {
		return float64(src.Pix[py*2+px])
	}
	xPos := [4]int{1, 0, 1, 0} // reflect101(-1,2), reflect101(0,2), reflect101(1,2), reflect101(2,2)
	yPos := [4]int{1, 0, 1, 0}
	expected := 0.0
	for ky := 0; ky < 4; ky++ {
		for kx := 0; kx < 4; kx++ {
			expected += w[ky] * w[kx] * vals(xPos[kx], yPos[ky])
		}
	}

	got := float64(dst.At(0, 0))
	if math.Abs(got-expected) > tol32 {
		t.Errorf("2x2→1x1: got %f, want %f", got, expected)
	}
}

func TestResizeCubicHalf_4x4(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 4, 4))
	for i := range src.Pix {
		src.Pix[i] = float32(i)
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	ResizeCubicHalf(dst, src)
	if dst.Width() != 2 || dst.Height() != 2 {
		t.Fatalf("output size %dx%d, want 2x2", dst.Width(), dst.Height())
	}

	// Hand-compute output pixel (0,0): center sx=0, sy=0
	w := cubicWeight
	expected00 := 0.0
	for ky := 0; ky < 4; ky++ {
		py := reflectBorder101(0-1+ky, 4) // positions: -1→1, 0→0, 1→1, 2→2
		for kx := 0; kx < 4; kx++ {
			px := reflectBorder101(0-1+kx, 4)
			expected00 += w[ky] * w[kx] * float64(src.Pix[py*4+px])
		}
	}
	got00 := float64(dst.At(0, 0))
	if math.Abs(got00-expected00) > tol32 {
		t.Errorf("(0,0): got %f, want %f", got00, expected00)
	}

	// Compute output pixel (1,1): center sx=2, sy=2
	expected11 := 0.0
	for ky := 0; ky < 4; ky++ {
		py := reflectBorder101(2-1+ky, 4) // positions: 1,2,3,2
		for kx := 0; kx < 4; kx++ {
			px := reflectBorder101(2-1+kx, 4)
			expected11 += w[ky] * w[kx] * float64(src.Pix[py*4+px])
		}
	}
	got11 := float64(dst.At(1, 1))
	if math.Abs(got11-expected11) > tol32 {
		t.Errorf("(1,1): got %f, want %f", got11, expected11)
	}
}

func TestResizeCubicHalf_OddSize(t *testing.T) {
	t.Parallel()
	src := imageutil.NewFloatImage(image.Rect(0, 0, 7, 7))
	for i := range src.Pix {
		src.Pix[i] = 1
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	ResizeCubicHalf(dst, src)
	if dst.Width() != 3 || dst.Height() != 3 {
		t.Errorf("7x7 → %dx%d, want 3x3", dst.Width(), dst.Height())
	}
}

func TestResizeCubicHalf_LinearGradient(t *testing.T) {
	t.Parallel()
	w, h := 20, 10
	src := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.Pix[y*w+x] = float32(x)
		}
	}
	dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
	ResizeCubicHalf(dst, src)

	// Each row of output should be monotonically non-decreasing
	dw := dst.Width()
	for y := 0; y < dst.Height(); y++ {
		for x := 1; x < dw; x++ {
			prev := dst.Pix[y*dst.Stride+x-1]
			curr := dst.Pix[y*dst.Stride+x]
			if curr < prev-tol32 {
				t.Errorf("row %d: not monotonic at x=%d (%f < %f)", y, x, curr, prev)
				break
			}
		}
	}
}

func TestResizeCubicHalf_EnergyPreservation(t *testing.T) {
	t.Parallel()
	for _, val := range []float32{0.0, 255.0} {
		src := imageutil.NewFloatImage(image.Rect(0, 0, 10, 10))
		for i := range src.Pix {
			src.Pix[i] = val
		}
		dst := imageutil.NewFloatImage(image.Rect(0, 0, 0, 0))
		ResizeCubicHalf(dst, src)
		for i, v := range dst.Pix {
			if math.Abs(float64(v)-float64(val)) > tol32 {
				t.Errorf("constant %.0f: Pix[%d] = %f", val, i, v)
				break
			}
		}
	}
}
