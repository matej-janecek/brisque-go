package imageutil

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// tol32 is the tolerance for float32 comparisons.
const tol32 = 1e-6

func TestNewFloatImage(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		rect  image.Rectangle
		wantW int
		wantH int
		wantS int
		wantN int
	}{
		{"zero", image.Rect(0, 0, 0, 0), 0, 0, 0, 0},
		{"1x1", image.Rect(0, 0, 1, 1), 1, 1, 1, 1},
		{"non-origin", image.Rect(10, 20, 50, 60), 40, 40, 40, 1600},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			f := NewFloatImage(tt.rect)
			if f.Width() != tt.wantW {
				t.Errorf("Width() = %d, want %d", f.Width(), tt.wantW)
			}
			if f.Height() != tt.wantH {
				t.Errorf("Height() = %d, want %d", f.Height(), tt.wantH)
			}
			if f.Stride != tt.wantS {
				t.Errorf("Stride = %d, want %d", f.Stride, tt.wantS)
			}
			if len(f.Pix) != tt.wantN {
				t.Errorf("len(Pix) = %d, want %d", len(f.Pix), tt.wantN)
			}
			if f.Rect != tt.rect {
				t.Errorf("Rect = %v, want %v", f.Rect, tt.rect)
			}
			// All zeros
			for i, v := range f.Pix {
				if v != 0 {
					t.Errorf("Pix[%d] = %f, want 0", i, v)
					break
				}
			}
		})
	}
}

func TestFloatImage_At(t *testing.T) {
	t.Parallel()

	t.Run("known_value", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 5, 5))
		f.Pix[3*5+2] = 42.5 // (2,3)
		if got := f.At(2, 3); got != 42.5 {
			t.Errorf("At(2,3) = %f, want 42.5", got)
		}
	})

	t.Run("oob_returns_zero", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 5, 5))
		f.Pix[0] = 99
		oob := [][2]int{{-1, 0}, {0, -1}, {5, 0}, {0, 5}}
		for _, p := range oob {
			if got := f.At(p[0], p[1]); got != 0 {
				t.Errorf("At(%d,%d) = %f, want 0", p[0], p[1], got)
			}
		}
	})

	t.Run("boundary", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 3, 3))
		f.Pix[0] = 1     // (0,0)
		f.Pix[2*3+2] = 2 // (2,2)
		if got := f.At(0, 0); got != 1 {
			t.Errorf("At(0,0) = %f, want 1", got)
		}
		if got := f.At(2, 2); got != 2 {
			t.Errorf("At(2,2) = %f, want 2", got)
		}
	})

	t.Run("non_origin", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(10, 20, 13, 23))
		f.Pix[0] = 77
		if got := f.At(10, 20); got != 77 {
			t.Errorf("At(10,20) = %f, want 77 (Pix[0])", got)
		}
		// OOB at origin
		if got := f.At(0, 0); got != 0 {
			t.Errorf("At(0,0) on non-origin image = %f, want 0", got)
		}
	})
}

func TestFloatImage_Set(t *testing.T) {
	t.Parallel()

	t.Run("set_and_read", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 4, 4))
		f.Set(1, 2, 3.14)
		if got := f.At(1, 2); math.Abs(float64(got)-3.14) > tol32 {
			t.Errorf("At(1,2) = %f, want 3.14", got)
		}
	})

	t.Run("oob_no_panic_no_mutation", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 3, 3))
		// Fill with sentinel
		for i := range f.Pix {
			f.Pix[i] = 1
		}
		// Set OOB — must not panic
		f.Set(-1, 0, 999)
		f.Set(3, 0, 999)
		f.Set(0, -1, 999)
		f.Set(0, 3, 999)
		// No pixel should have changed
		for i, v := range f.Pix {
			if v != 1 {
				t.Errorf("Pix[%d] = %f after OOB set, want 1", i, v)
			}
		}
	})

	t.Run("boundary", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 3, 3))
		f.Set(0, 0, 10)
		f.Set(2, 2, 20)
		if f.At(0, 0) != 10 || f.At(2, 2) != 20 {
			t.Error("boundary set/get mismatch")
		}
	})

	t.Run("special_floats", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 3, 1))
		f.Set(0, 0, float32(math.NaN()))
		f.Set(1, 0, float32(math.Inf(1)))
		f.Set(2, 0, float32(math.Inf(-1)))
		if !math.IsNaN(float64(f.At(0, 0))) {
			t.Error("NaN not stored")
		}
		if !math.IsInf(float64(f.At(1, 0)), 1) {
			t.Error("+Inf not stored")
		}
		if !math.IsInf(float64(f.At(2, 0)), -1) {
			t.Error("-Inf not stored")
		}
	})
}

func TestFloatImage_SubImage(t *testing.T) {
	t.Parallel()

	t.Run("full_rect", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 4, 4))
		for i := range f.Pix {
			f.Pix[i] = float32(i)
		}
		sub := f.SubImage(f.Rect)
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				if sub.At(x, y) != f.At(x, y) {
					t.Errorf("SubImage At(%d,%d) = %f, parent = %f", x, y, sub.At(x, y), f.At(x, y))
				}
			}
		}
	})

	t.Run("smaller_region", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 4, 4))
		for i := range f.Pix {
			f.Pix[i] = float32(i)
		}
		sub := f.SubImage(image.Rect(1, 1, 3, 3))
		if sub.Width() != 2 || sub.Height() != 2 {
			t.Fatalf("sub size %dx%d, want 2x2", sub.Width(), sub.Height())
		}
		// sub.At(1,1) should be parent's Pix[1*4+1] = 5
		if got := sub.At(1, 1); got != 5 {
			t.Errorf("sub.At(1,1) = %f, want 5", got)
		}
	})

	t.Run("mutation_propagates", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 4, 4))
		sub := f.SubImage(image.Rect(1, 1, 3, 3))
		sub.Set(2, 2, 99)
		if f.At(2, 2) != 99 {
			t.Errorf("parent.At(2,2) = %f after sub.Set, want 99", f.At(2, 2))
		}
	})

	t.Run("non_overlapping", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 4, 4))
		sub := f.SubImage(image.Rect(10, 10, 20, 20))
		if !sub.Rect.Empty() {
			t.Errorf("non-overlapping SubImage should be empty, got %v", sub.Rect)
		}
	})

	t.Run("partial_overlap", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 4, 4))
		sub := f.SubImage(image.Rect(2, 2, 10, 10))
		want := image.Rect(2, 2, 4, 4)
		if sub.Rect != want {
			t.Errorf("partial overlap Rect = %v, want %v", sub.Rect, want)
		}
	})

	t.Run("stride_equals_parent", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 8, 8))
		sub := f.SubImage(image.Rect(1, 1, 4, 4))
		if sub.Stride != f.Stride {
			t.Errorf("sub.Stride = %d, want parent Stride %d", sub.Stride, f.Stride)
		}
	})
}

func TestFloatImage_Reset(t *testing.T) {
	t.Parallel()

	t.Run("smaller_reuse", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 10, 10))
		oldCap := cap(f.Pix)
		f.Pix[0] = 99
		f.Reset(image.Rect(0, 0, 3, 3))
		if f.Width() != 3 || f.Height() != 3 {
			t.Errorf("after Reset: %dx%d, want 3x3", f.Width(), f.Height())
		}
		if cap(f.Pix) != oldCap {
			t.Error("buffer was reallocated for smaller reset")
		}
		for i, v := range f.Pix {
			if v != 0 {
				t.Errorf("Pix[%d] = %f after Reset, want 0", i, v)
				break
			}
		}
	})

	t.Run("larger_alloc", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 2, 2))
		f.Reset(image.Rect(0, 0, 20, 20))
		if len(f.Pix) != 400 {
			t.Errorf("len(Pix) = %d after larger Reset, want 400", len(f.Pix))
		}
		for i, v := range f.Pix {
			if v != 0 {
				t.Errorf("Pix[%d] = %f, want 0", i, v)
				break
			}
		}
	})

	t.Run("same_size_no_stale", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 5, 5))
		for i := range f.Pix {
			f.Pix[i] = float32(i)
		}
		f.Reset(image.Rect(0, 0, 5, 5))
		for i, v := range f.Pix {
			if v != 0 {
				t.Errorf("Pix[%d] = %f after same-size Reset, want 0", i, v)
				break
			}
		}
	})

	t.Run("zero_size", func(t *testing.T) {
		t.Parallel()
		f := NewFloatImage(image.Rect(0, 0, 10, 10))
		f.Reset(image.Rect(0, 0, 0, 0))
		if len(f.Pix) != 0 {
			t.Errorf("len(Pix) = %d after zero-size Reset, want 0", len(f.Pix))
		}
	})
}

func TestFromGrayBytesInto(t *testing.T) {
	t.Parallel()

	t.Run("known_values", func(t *testing.T) {
		t.Parallel()
		pix := []byte{0, 128, 255}
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromGrayBytesInto(dst, pix, 3, 1)
		want := []float32{0, 128, 255}
		for i, w := range want {
			if dst.Pix[i] != w {
				t.Errorf("Pix[%d] = %f, want %f", i, dst.Pix[i], w)
			}
		}
	})

	t.Run("1x1", func(t *testing.T) {
		t.Parallel()
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromGrayBytesInto(dst, []byte{200}, 1, 1)
		if dst.At(0, 0) != 200 {
			t.Errorf("1x1: At(0,0) = %f, want 200", dst.At(0, 0))
		}
	})

	t.Run("row_major_ordering", func(t *testing.T) {
		t.Parallel()
		// 2×2 image: pixel at (1,0)=10, (0,1)=20
		pix := []byte{0, 10, 20, 30}
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromGrayBytesInto(dst, pix, 2, 2)
		if got := dst.At(1, 0); got != 10 {
			t.Errorf("At(1,0) = %f, want 10", got)
		}
		if got := dst.At(0, 1); got != 20 {
			t.Errorf("At(0,1) = %f, want 20", got)
		}
		if dst.Rect != image.Rect(0, 0, 2, 2) {
			t.Errorf("Rect = %v, want (0,0)-(2,2)", dst.Rect)
		}
	})
}

func TestFromImageInto(t *testing.T) {
	t.Parallel()

	t.Run("gray", func(t *testing.T) {
		t.Parallel()
		img := image.NewGray(image.Rect(0, 0, 2, 1))
		img.Pix[0] = 200
		img.Pix[1] = 0
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromImageInto(dst, img)
		if got := dst.At(0, 0); got != 200 {
			t.Errorf("gray byte 200 → %f, want 200", got)
		}
		if got := dst.At(1, 0); got != 0 {
			t.Errorf("gray byte 0 → %f, want 0", got)
		}
	})

	t.Run("rgba_colors", func(t *testing.T) {
		t.Parallel()
		tests := []struct {
			name string
			c    color.RGBA
			want float64
		}{
			{"red", color.RGBA{255, 0, 0, 255}, 0.299 * 255},
			{"green", color.RGBA{0, 255, 0, 255}, 0.587 * 255},
			{"white", color.RGBA{255, 255, 255, 255}, 255.0},
			{"black", color.RGBA{0, 0, 0, 255}, 0.0},
		}
		for _, tt := range tests {
			tt := tt
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()
				img := image.NewRGBA(image.Rect(0, 0, 1, 1))
				img.SetRGBA(0, 0, tt.c)
				dst := NewFloatImage(image.Rect(0, 0, 0, 0))
				FromImageInto(dst, img)
				got := float64(dst.At(0, 0))
				if math.Abs(got-tt.want) > 0.01 {
					t.Errorf("%s: got %f, want %f", tt.name, got, tt.want)
				}
			})
		}
	})

	t.Run("nrgba_alpha_zero", func(t *testing.T) {
		t.Parallel()
		img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		img.SetNRGBA(0, 0, color.NRGBA{R: 200, G: 100, B: 50, A: 0})
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromImageInto(dst, img)
		if got := dst.At(0, 0); got != 0 {
			t.Errorf("NRGBA alpha=0: got %f, want 0", got)
		}
	})

	t.Run("nrgba_alpha_128", func(t *testing.T) {
		t.Parallel()
		img := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		img.SetNRGBA(0, 0, color.NRGBA{R: 200, G: 100, B: 50, A: 128})
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromImageInto(dst, img)
		a := 128.0 / 255.0
		rEff := 200.0 * a
		gEff := 100.0 * a
		bEff := 50.0 * a
		want := 0.299*rEff + 0.587*gEff + 0.114*bEff
		got := float64(dst.At(0, 0))
		if math.Abs(got-want) > 0.01 {
			t.Errorf("NRGBA alpha=128: got %f, want %f", got, want)
		}
	})

	t.Run("ycbcr", func(t *testing.T) {
		t.Parallel()
		// Create YCbCr with known Y
		img := image.NewYCbCr(image.Rect(0, 0, 2, 2), image.YCbCrSubsampleRatio444)
		img.Y[0] = 180
		img.Y[1] = 50
		img.Y[2] = 0
		img.Y[3] = 255
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromImageInto(dst, img)
		if got := dst.At(0, 0); got != 180 {
			t.Errorf("YCbCr Y=180 → %f, want 180", got)
		}
		if got := dst.At(1, 0); got != 50 {
			t.Errorf("YCbCr Y=50 → %f, want 50", got)
		}
	})

	t.Run("generic_fallback", func(t *testing.T) {
		t.Parallel()
		// Custom image type that forces the generic path
		img := &genericImage{
			bounds: image.Rect(0, 0, 1, 1),
			c:      color.RGBA{100, 150, 200, 255},
		}
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromImageInto(dst, img)
		// Generic path uses RGBA() returning [0, 0xFFFF], then luminance()
		r, g, b, _ := img.c.RGBA()
		want := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 257.0
		got := float64(dst.At(0, 0))
		if math.Abs(got-want) > 0.01 {
			t.Errorf("generic: got %f, want %f", got, want)
		}
	})

	t.Run("non_origin_bounds", func(t *testing.T) {
		t.Parallel()
		img := image.NewGray(image.Rect(5, 5, 8, 8))
		img.SetGray(5, 5, color.Gray{Y: 42})
		img.SetGray(7, 7, color.Gray{Y: 99})
		dst := NewFloatImage(image.Rect(0, 0, 0, 0))
		FromImageInto(dst, img)
		if dst.Rect != image.Rect(5, 5, 8, 8) {
			t.Errorf("Rect = %v, want (5,5)-(8,8)", dst.Rect)
		}
		if got := dst.At(5, 5); got != 42 {
			t.Errorf("At(5,5) = %f, want 42", got)
		}
		if got := dst.At(7, 7); got != 99 {
			t.Errorf("At(7,7) = %f, want 99", got)
		}
	})

	t.Run("gray_equivalence_across_types", func(t *testing.T) {
		t.Parallel()
		// R=G=B=V should produce the same luminance across all typed paths
		v := uint8(123)
		grayImg := image.NewGray(image.Rect(0, 0, 1, 1))
		grayImg.Pix[0] = v

		rgbaImg := image.NewRGBA(image.Rect(0, 0, 1, 1))
		rgbaImg.SetRGBA(0, 0, color.RGBA{v, v, v, 255})

		nrgbaImg := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		nrgbaImg.SetNRGBA(0, 0, color.NRGBA{v, v, v, 255})

		dst := NewFloatImage(image.Rect(0, 0, 0, 0))

		FromImageInto(dst, grayImg)
		grayVal := dst.At(0, 0)

		FromImageInto(dst, rgbaImg)
		rgbaVal := dst.At(0, 0)

		FromImageInto(dst, nrgbaImg)
		nrgbaVal := dst.At(0, 0)

		// Gray gives exactly v; RGBA gives 0.299*v + 0.587*v + 0.114*v = v
		if math.Abs(float64(grayVal)-float64(rgbaVal)) > 0.01 {
			t.Errorf("gray(%f) != rgba(%f) for V=%d", grayVal, rgbaVal, v)
		}
		if math.Abs(float64(grayVal)-float64(nrgbaVal)) > 0.01 {
			t.Errorf("gray(%f) != nrgba(%f) for V=%d", grayVal, nrgbaVal, v)
		}
	})
}

// genericImage is a custom image type that forces the fromGeneric path.
type genericImage struct {
	bounds image.Rectangle
	c      color.Color
}

func (g *genericImage) ColorModel() color.Model { return color.RGBAModel }
func (g *genericImage) Bounds() image.Rectangle { return g.bounds }
func (g *genericImage) At(x, y int) color.Color {
	if !(image.Point{X: x, Y: y}).In(g.bounds) {
		return color.RGBA{}
	}
	return g.c
}
