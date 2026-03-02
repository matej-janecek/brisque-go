package brisque

import (
	"context"
	"image"
	"image/color"
	"image/jpeg"
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
