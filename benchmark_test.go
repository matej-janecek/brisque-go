package brisque

import (
	"context"
	"image"
	"math/rand"
	"testing"
)

func makeGrayImage(w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	rng := rand.New(rand.NewSource(42))
	for i := range img.Pix {
		img.Pix[i] = uint8(rng.Intn(256))
	}
	return img
}

func BenchmarkScoreImage_1080p(b *testing.B) {
	m := DefaultModel()
	ctx := context.Background()
	img := makeGrayImage(1920, 1080)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = m.ScoreImage(ctx, img)
	}
}

func BenchmarkScoreImage_4K(b *testing.B) {
	m := DefaultModel()
	ctx := context.Background()
	img := makeGrayImage(3840, 2160)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = m.ScoreImage(ctx, img)
	}
}

func BenchmarkScoreGray_1080p(b *testing.B) {
	m := DefaultModel()
	ctx := context.Background()
	w, h := 1920, 1080
	pix := make([]byte, w*h)
	rng := rand.New(rand.NewSource(42))
	for i := range pix {
		pix[i] = uint8(rng.Intn(256))
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = m.ScoreGray(ctx, pix, w, h)
	}
}

func BenchmarkScoreWithWorkspace_1080p(b *testing.B) {
	m := DefaultModel()
	ctx := context.Background()
	img := makeGrayImage(1920, 1080)
	ws := NewWorkspace(1920, 1080)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = m.ScoreWithWorkspace(ctx, ws, img)
	}
}

func BenchmarkScoreBatch_10Frames(b *testing.B) {
	m := DefaultModel()
	ctx := context.Background()
	images := make([]image.Image, 10)
	for i := range images {
		images[i] = makeGrayImage(1920, 1080)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = m.ScoreBatch(ctx, images)
	}
}

func BenchmarkConvolution_1080p(b *testing.B) {
	m := DefaultModel()
	ctx := context.Background()
	img := makeGrayImage(1920, 1080)

	// Just measure a single score which includes convolution
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = m.ScoreImage(ctx, img)
	}
}
