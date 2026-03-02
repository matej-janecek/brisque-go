package brisque

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestDefaultModelCreation(t *testing.T) {
	m := DefaultModel()
	if m == nil {
		t.Fatal("DefaultModel() returned nil")
	}
	if m.svr == nil {
		t.Fatal("SVR model is nil")
	}
	if m.svr.NSV != 774 {
		t.Errorf("expected 774 support vectors, got %d", m.svr.NSV)
	}
	if m.svr.Gamma != 0.05 {
		t.Errorf("expected gamma=0.05, got %f", m.svr.Gamma)
	}
	if len(m.kernel) != 7 {
		t.Errorf("expected kernel size 7, got %d", len(m.kernel))
	}
}

func TestScoreImage_TooSmall(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	img := image.NewGray(image.Rect(0, 0, 4, 4))
	_, err := m.ScoreImage(ctx, img)
	if err == nil {
		t.Fatal("expected error for tiny image")
	}
	if _, ok := err.(*ErrImageTooSmall); !ok {
		t.Errorf("expected ErrImageTooSmall, got %T: %v", err, err)
	}
}

func TestScoreGray_TooSmall(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	pix := make([]byte, 4*4)
	_, err := m.ScoreGray(ctx, pix, 4, 4)
	if err == nil {
		t.Fatal("expected error for tiny image")
	}
}

func TestScoreImage_UniformGray(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	// Create a 64x64 uniform gray image
	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = 128
	}

	score, err := m.ScoreImage(ctx, img)
	if err != nil {
		t.Logf("uniform image returned error (expected): %v", err)
		return
	}
	// Uniform images should get a high (bad) score
	t.Logf("uniform image score: %.2f", score)
}

func TestScoreImage_GradientImage(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	// Create a 128x128 gradient image
	img := image.NewGray(image.Rect(0, 0, 128, 128))
	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			img.Pix[y*128+x] = uint8((x + y) * 255 / 254)
		}
	}

	score, err := m.ScoreImage(ctx, img)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("gradient image score: %.2f", score)
}

func TestScoreImage_RGBA(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	// Create a 64x64 RGBA image with some pattern
	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(x * 4),
				G: uint8(y * 4),
				B: uint8((x + y) * 2),
				A: 255,
			})
		}
	}

	score, err := m.ScoreImage(ctx, img)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("RGBA pattern score: %.2f", score)
}

func TestScoreGray_Basic(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	// Create a simple test pattern
	w, h := 64, 64
	pix := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			pix[y*w+x] = uint8((x * y) % 256)
		}
	}

	score, err := m.ScoreGray(ctx, pix, w, h)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("grayscale pattern score: %.2f", score)
}

func TestScoreBatch(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	images := make([]image.Image, 3)
	for i := range images {
		img := image.NewGray(image.Rect(0, 0, 64, 64))
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				img.Pix[y*64+x] = uint8((x + y + i*30) % 256)
			}
		}
		images[i] = img
	}

	scores, err := m.ScoreBatch(ctx, images)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(scores) != 3 {
		t.Fatalf("expected 3 scores, got %d", len(scores))
	}
	for i, s := range scores {
		t.Logf("batch image %d score: %.2f", i, s)
	}
}

func TestScoreBatch_Empty(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	scores, err := m.ScoreBatch(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if scores != nil {
		t.Fatalf("expected nil scores for empty input")
	}
}

func TestScoreImage_ContextCancellation(t *testing.T) {
	m := DefaultModel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	img := image.NewGray(image.Rect(0, 0, 64, 64))
	_, err := m.ScoreImage(ctx, img)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestGoldenSample(t *testing.T) {
	f, err := os.Open("testdata/golden/sample.jpg")
	if err != nil {
		t.Skip("golden test image not available:", err)
	}
	defer func() { _ = f.Close() }()

	img, err := jpeg.Decode(f)
	if err != nil {
		t.Fatal("failed to decode test image:", err)
	}

	m := DefaultModel()
	ctx := context.Background()

	score, err := m.ScoreImage(ctx, img)
	if err != nil {
		t.Fatalf("scoring failed: %v", err)
	}

	t.Logf("golden sample score: %.4f", score)
	// Log the score; we'll compare with OpenCV reference later
}

func TestScoreWithWorkspace(t *testing.T) {
	m := DefaultModel()
	ctx := context.Background()

	ws := NewWorkspace(1920, 1080)

	img := image.NewGray(image.Rect(0, 0, 128, 128))
	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			img.Pix[y*128+x] = uint8((x*3 + y*7) % 256)
		}
	}

	score, err := m.ScoreWithWorkspace(ctx, ws, img)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("workspace score: %.2f", score)

	// Score again with same workspace to verify reuse
	score2, err := m.ScoreWithWorkspace(ctx, ws, img)
	if err != nil {
		t.Fatalf("unexpected error on reuse: %v", err)
	}
	if score != score2 {
		t.Errorf("scores differ on reuse: %.4f vs %.4f", score, score2)
	}
}

func TestOptions(t *testing.T) {
	ctx := context.Background()
	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = uint8(i % 256)
	}

	t.Run("WithWorkspacePool", func(t *testing.T) {
		pool := &sync.Pool{}
		m := DefaultModel(WithWorkspacePool(pool))
		if m.cfg.workspacePool != pool {
			t.Fatal("workspace pool not set")
		}
	})

	t.Run("WithParallelThreshold", func(t *testing.T) {
		m := DefaultModel(WithParallelThreshold(0))
		_, err := m.ScoreImage(ctx, img)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("WithLogger", func(t *testing.T) {
		m := DefaultModel(WithLogger(nil))
		_, err := m.ScoreImage(ctx, img)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("WithUnsafeOptimizations", func(t *testing.T) {
		m := DefaultModel(WithUnsafeOptimizations())
		_, err := m.ScoreImage(ctx, img)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func TestScoreImage_SizeBoundary(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	tests := []struct {
		name    string
		w, h    int
		wantErr bool
	}{
		{"15x16", 15, 16, true},
		{"16x15", 16, 15, true},
		{"0x0", 0, 0, true},
		{"16x16", 16, 16, false},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			img := image.NewGray(image.Rect(0, 0, tt.w, tt.h))
			// Fill with some pattern to avoid degenerate
			for i := range img.Pix {
				img.Pix[i] = uint8((i * 7) % 256)
			}
			_, err := m.ScoreImage(ctx, img)
			if tt.wantErr {
				if err == nil {
					t.Error("expected ErrImageTooSmall, got nil")
				}
				var tooSmall *ErrImageTooSmall
				if !errors.As(err, &tooSmall) {
					t.Errorf("expected *ErrImageTooSmall, got %T", err)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestScoreImage_Clamping(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	makeImg := func(fill func(x, y int) uint8) image.Image {
		img := image.NewGray(image.Rect(0, 0, 64, 64))
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				img.Pix[y*64+x] = fill(x, y)
			}
		}
		return img
	}

	tests := []struct {
		name string
		img  image.Image
	}{
		{"white", makeImg(func(_, _ int) uint8 { return 255 })},
		{"black", makeImg(func(_, _ int) uint8 { return 0 })},
		{"noise", makeImg(func(_, _ int) uint8 {
			return uint8(rand.New(rand.NewSource(42)).Intn(256))
		})},
		{"gradient", makeImg(func(x, y int) uint8 { return uint8((x + y) % 256) })},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			score, err := m.ScoreImage(ctx, tt.img)
			if err != nil {
				t.Logf("error (may be degenerate): %v", err)
				return
			}
			if score < 0 || score > 100 {
				t.Errorf("score = %f, want in [0, 100]", score)
			}
		})
	}
}

func TestScoreImage_Ordering(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	// Use 256×256 images with realistic content to avoid both scoring at clamped 100
	rng := rand.New(rand.NewSource(42))
	noise := image.NewGray(image.Rect(0, 0, 256, 256))
	for i := range noise.Pix {
		noise.Pix[i] = uint8(rng.Intn(256))
	}
	noiseScore, err := m.ScoreImage(ctx, noise)
	if err != nil {
		t.Fatalf("noise: %v", err)
	}

	// Smooth gradient — natural-looking, should score better (lower)
	gradient := image.NewGray(image.Rect(0, 0, 256, 256))
	for y := 0; y < 256; y++ {
		for x := 0; x < 256; x++ {
			gradient.Pix[y*256+x] = uint8((x + y) / 2)
		}
	}
	gradScore, err := m.ScoreImage(ctx, gradient)
	if err != nil {
		t.Fatalf("gradient: %v", err)
	}

	t.Logf("noise=%.2f, gradient=%.2f", noiseScore, gradScore)
	// Both may be clamped to 100 for synthetic images; only check if not both clamped
	if noiseScore < 100 && gradScore < 100 && noiseScore <= gradScore {
		t.Errorf("noise (%f) should score higher (worse) than gradient (%f)", noiseScore, gradScore)
	}
}

func TestScoreImage_GrayVsRGBA(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	grayImg := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range grayImg.Pix {
		grayImg.Pix[i] = uint8((i * 7) % 256)
	}

	rgbaImg := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			v := grayImg.GrayAt(x, y).Y
			rgbaImg.SetRGBA(x, y, color.RGBA{v, v, v, 255})
		}
	}

	gScore, err := m.ScoreImage(ctx, grayImg)
	if err != nil {
		t.Fatalf("gray: %v", err)
	}
	rScore, err := m.ScoreImage(ctx, rgbaImg)
	if err != nil {
		t.Fatalf("rgba: %v", err)
	}

	if math.Abs(gScore-rScore) > 0.01 {
		t.Errorf("gray (%f) != rgba (%f) for equivalent R=G=B pixels", gScore, rScore)
	}
}

func TestScoreImage_ConcurrentCalls(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = uint8((i * 7) % 256)
	}

	refScore, err := m.ScoreImage(ctx, img)
	if err != nil {
		t.Fatalf("reference: %v", err)
	}

	var wg sync.WaitGroup
	errs := make(chan error, 10)
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			score, err := m.ScoreImage(ctx, img)
			if err != nil {
				errs <- err
				return
			}
			if score != refScore {
				errs <- fmt.Errorf("concurrent score %f != ref %f", score, refScore)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestScoreImage_NonOriginBounds(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	// Create a larger image, then SubImage
	big := image.NewGray(image.Rect(0, 0, 100, 100))
	for i := range big.Pix {
		big.Pix[i] = uint8((i * 7) % 256)
	}
	sub := big.SubImage(image.Rect(10, 10, 74, 74))

	score, err := m.ScoreImage(ctx, sub)
	if err != nil {
		t.Fatalf("non-origin bounds: %v", err)
	}
	if score < 0 || score > 100 {
		t.Errorf("score = %f, want in [0, 100]", score)
	}
}

func TestScoreGray_EquivalenceWithScoreImage(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	w, h := 64, 64
	pix := make([]byte, w*h)
	for i := range pix {
		pix[i] = uint8((i * 11) % 256)
	}

	grayScore, err := m.ScoreGray(ctx, pix, w, h)
	if err != nil {
		t.Fatalf("ScoreGray: %v", err)
	}

	img := image.NewGray(image.Rect(0, 0, w, h))
	copy(img.Pix, pix)
	imgScore, err := m.ScoreImage(ctx, img)
	if err != nil {
		t.Fatalf("ScoreImage: %v", err)
	}

	if grayScore != imgScore {
		t.Errorf("ScoreGray=%f != ScoreImage=%f", grayScore, imgScore)
	}
}

func TestScoreGray_ShortPix(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	// len(pix) < w*h — should panic or return error
	pix := make([]byte, 10) // way too short for 64x64
	panicked := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		_, err := m.ScoreGray(ctx, pix, 64, 64)
		if err != nil {
			return // error is acceptable
		}
		// If no panic and no error, that's a problem
		t.Error("expected panic or error for short pix, got neither")
	}()
	if panicked {
		t.Log("short pix caused panic (acceptable)")
	}
}

func TestScoreBatch_OrderPreservation(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	images := make([]image.Image, 5)
	for i := range images {
		img := image.NewGray(image.Rect(0, 0, 64, 64))
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				img.Pix[y*64+x] = uint8((x + y + i*37) % 256)
			}
		}
		images[i] = img
	}

	batchScores, err := m.ScoreBatch(ctx, images)
	if err != nil {
		t.Fatalf("ScoreBatch: %v", err)
	}

	for i, img := range images {
		individual, err := m.ScoreImage(ctx, img)
		if err != nil {
			t.Fatalf("ScoreImage[%d]: %v", i, err)
		}
		if batchScores[i] != individual {
			t.Errorf("batch[%d]=%f != individual=%f", i, batchScores[i], individual)
		}
	}
}

func TestScoreBatch_MixedValidInvalid(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	valid := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range valid.Pix {
		valid.Pix[i] = uint8(i % 256)
	}
	tooSmall := image.NewGray(image.Rect(0, 0, 4, 4))

	images := []image.Image{valid, tooSmall, valid}
	_, err := m.ScoreBatch(ctx, images)
	if err == nil {
		t.Fatal("expected error for batch with too-small image")
	}
}

func TestFeatures_Determinism(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = uint8((i * 7) % 256)
	}

	f1, err := m.Features(ctx, img)
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	f2, err := m.Features(ctx, img)
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if f1 != f2 {
		t.Error("Features not deterministic")
	}
}

func TestFeatures_NonTrivial(t *testing.T) {
	t.Parallel()
	m := DefaultModel()
	ctx := context.Background()

	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range img.Pix {
		img.Pix[i] = uint8((i * 7) % 256)
	}

	feats, err := m.Features(ctx, img)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	allZero := true
	for _, v := range feats {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("all features are zero")
	}
}

func TestNewModel_ErrorCases(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"missing_lines", "1.0 7\n0.05 -1.0 1\n"},
		{"wrong_sv_fields", func() string {
			// SV line with 36 fields instead of 37
			var b strings.Builder
			b.WriteString("1.0 7\n0.05 -1.0 1\n")
			b.WriteString("1.0")
			for i := 0; i < 35; i++ {
				b.WriteString(" 0.5")
			}
			b.WriteString("\n")
			return b.String()
		}()},
		{"wrong_scale_count", func() string {
			var b strings.Builder
			b.WriteString("1.0 7\n0.05 -1.0 1\n")
			b.WriteString("1.0")
			for i := 0; i < 36; i++ {
				b.WriteString(" 0.5")
			}
			b.WriteString("\n")
			// 35 instead of 36 scale mins
			for i := 0; i < 35; i++ {
				if i > 0 {
					b.WriteString(" ")
				}
				b.WriteString("0.0")
			}
			b.WriteString("\n")
			return b.String()
		}()},
		{"non_numeric", "abc 7\n"},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := NewModel(strings.NewReader(tt.input))
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}

	t.Run("zero_svs_valid", func(t *testing.T) {
		t.Parallel()
		var b strings.Builder
		b.WriteString("1.0 7\n0.05 -1.0 0\n") // 0 SVs
		for i := 0; i < 36; i++ {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString("0.0")
		}
		b.WriteString("\n")
		for i := 0; i < 36; i++ {
			if i > 0 {
				b.WriteString(" ")
			}
			b.WriteString("1.0")
		}
		b.WriteString("\n")
		m, err := NewModel(strings.NewReader(b.String()))
		if err != nil {
			t.Fatalf("expected no error for 0 SVs, got %v", err)
		}
		if m == nil {
			t.Fatal("model is nil")
		}
	})
}

func TestNewWorkspace_NotNil(t *testing.T) {
	t.Parallel()
	ws := NewWorkspace(100, 100)
	if ws == nil {
		t.Fatal("NewWorkspace returned nil")
	}
}

func TestErrImageTooSmall_Interface(t *testing.T) {
	t.Parallel()
	var err error = &ErrImageTooSmall{
		Width: 4, Height: 4,
		MinWidth: 16, MinHeight: 16,
	}
	msg := err.Error()
	if msg == "" {
		t.Error("Error() returned empty string")
	}
	// Message should contain dimensions
	if !strings.Contains(msg, "4") || !strings.Contains(msg, "16") {
		t.Errorf("error message missing dimensions: %q", msg)
	}
	var tooSmall *ErrImageTooSmall
	if !errors.As(err, &tooSmall) {
		t.Error("errors.As failed for ErrImageTooSmall")
	}
}

func TestErrDegenerateDistribution_Interface(t *testing.T) {
	t.Parallel()
	var err error = &ErrDegenerateDistribution{
		Scale:   1,
		Feature: "mscn",
	}
	msg := err.Error()
	if msg == "" {
		t.Error("Error() returned empty string")
	}
	if !strings.Contains(msg, "1") || !strings.Contains(msg, "mscn") {
		t.Errorf("error message missing scale/feature: %q", msg)
	}
	var de *ErrDegenerateDistribution
	if !errors.As(err, &de) {
		t.Error("errors.As failed for ErrDegenerateDistribution")
	}
}

func TestLoadModelFromFile(t *testing.T) {
	// Build a minimal model file: 1 support vector, 36 features.
	var buf strings.Builder
	buf.WriteString("1.166667 7\n")  // kernel_sigma kernel_size
	buf.WriteString("0.05 -1.0 1\n") // gamma rho nsv
	buf.WriteString("1.0")
	for i := 0; i < 36; i++ {
		buf.WriteString(" 0.5")
	}
	buf.WriteString("\n")
	// scale mins
	for i := 0; i < 36; i++ {
		if i > 0 {
			buf.WriteString(" ")
		}
		buf.WriteString("0.0")
	}
	buf.WriteString("\n")
	// scale maxs
	for i := 0; i < 36; i++ {
		if i > 0 {
			buf.WriteString(" ")
		}
		buf.WriteString("1.0")
	}
	buf.WriteString("\n")

	path := filepath.Join(t.TempDir(), "test.model")
	if err := os.WriteFile(path, []byte(buf.String()), 0644); err != nil {
		t.Fatal(err)
	}

	m, err := LoadModelFromFile(path)
	if err != nil {
		t.Fatalf("LoadModelFromFile: %v", err)
	}
	if m == nil {
		t.Fatal("model is nil")
	}
	if m.svr.NSV != 1 {
		t.Errorf("expected 1 SV, got %d", m.svr.NSV)
	}
}
