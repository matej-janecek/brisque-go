package features

import (
	"context"
	"errors"
	"image"
	"math/rand"
	"testing"

	"github.com/matej-janecek/brisque-go/internal/conv"
	"github.com/matej-janecek/brisque-go/internal/imageutil"
)

func makeKernel() []float64 {
	return conv.MakeGaussianKernel(7.0/6.0, 7)
}

func makeGradientImage(w, h int) *imageutil.FloatImage {
	img := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Pix[y*w+x] = float32((x + y) % 256)
		}
	}
	return img
}

func makeNoiseImage(w, h int) *imageutil.FloatImage {
	rng := rand.New(rand.NewSource(42))
	img := imageutil.NewFloatImage(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = float32(rng.Float64() * 255)
	}
	return img
}

func TestExtract_ConstantImage(t *testing.T) {
	t.Parallel()
	img := imageutil.NewFloatImage(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = 128
	}
	kernel := makeKernel()
	ws := NewWorkspace(64, 64)
	_, err := Extract(context.Background(), img, kernel, ws)
	// Constant image → MSCN ≈ 0 → either valid features or degenerate error
	// Both outcomes are acceptable
	if err != nil {
		t.Logf("constant image returned error (acceptable): %v", err)
	}
}

func TestExtract_TooSmall(t *testing.T) {
	t.Parallel()
	kernel := makeKernel() // size=7
	tests := []struct {
		name string
		w, h int
	}{
		{"3x3", 3, 3},
		{"7x3", 7, 3},
		{"3x7", 3, 7},
		// 7x7 image: original scale OK, but half=3×3 < 7 → TooSmallError at scale 2
		{"7x7_half_too_small", 7, 7},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			img := imageutil.NewFloatImage(image.Rect(0, 0, tt.w, tt.h))
			for i := range img.Pix {
				img.Pix[i] = float32(i % 256)
			}
			ws := NewWorkspace(tt.w, tt.h)
			_, err := Extract(context.Background(), img, kernel, ws)
			if err == nil {
				t.Fatal("expected TooSmallError, got nil")
			}
			var tse *TooSmallError
			if !errors.As(err, &tse) {
				t.Errorf("expected *TooSmallError, got %T: %v", err, err)
			}
		})
	}
}

func TestExtract_CancelledContext(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	img := makeGradientImage(64, 64)
	kernel := makeKernel()
	ws := NewWorkspace(64, 64)
	_, err := Extract(ctx, img, kernel, ws)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestExtract_FeatureCount(t *testing.T) {
	t.Parallel()
	img := makeGradientImage(64, 64)
	kernel := makeKernel()
	ws := NewWorkspace(64, 64)
	feats, err := Extract(context.Background(), img, kernel, ws)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Not all zero
	allZero := true
	for _, v := range feats {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("all features are zero for gradient image")
	}
}

func TestExtract_Determinism(t *testing.T) {
	t.Parallel()
	img := makeGradientImage(64, 64)
	kernel := makeKernel()

	ws1 := NewWorkspace(64, 64)
	f1, err := Extract(context.Background(), img, kernel, ws1)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}

	ws2 := NewWorkspace(64, 64)
	f2, err := Extract(context.Background(), img, kernel, ws2)
	if err != nil {
		t.Fatalf("second call: %v", err)
	}

	if f1 != f2 {
		t.Error("features differ between two identical calls")
		for i := range f1 {
			if f1[i] != f2[i] {
				t.Logf("  [%d]: %f vs %f", i, f1[i], f2[i])
			}
		}
	}
}

func TestExtract_Scale2DiffersFromScale1(t *testing.T) {
	t.Parallel()
	img := makeGradientImage(64, 64)
	kernel := makeKernel()
	ws := NewWorkspace(64, 64)
	feats, err := Extract(context.Background(), img, kernel, ws)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	same := true
	for i := 0; i < 18; i++ {
		if feats[i] != feats[i+18] {
			same = false
			break
		}
	}
	if same {
		t.Error("scale 1 features identical to scale 2 for non-self-similar image")
	}
}

func TestExtract_FeatureRanges(t *testing.T) {
	t.Parallel()
	img := makeNoiseImage(64, 64)
	kernel := makeKernel()
	ws := NewWorkspace(64, 64)
	feats, err := Extract(context.Background(), img, kernel, ws)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// For each scale, features[0] is AGGD alpha → should be in [0.2, 10]
	// features[1] is (lσ²+rσ²)/2 → should be ≥ 0
	for _, base := range []int{0, 18} {
		alpha := feats[base]
		if alpha < 0.2 || alpha > 10.0 {
			t.Errorf("alpha at offset %d = %f, want in [0.2, 10]", base, alpha)
		}
		avgSigma := feats[base+1]
		if avgSigma < 0 {
			t.Errorf("avg sigma² at offset %d = %f, want >= 0", base+1, avgSigma)
		}
	}
}
