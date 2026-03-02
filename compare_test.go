//go:build integration

package brisque

import (
	"context"
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/matej/brisque-go/internal/features"
	"github.com/matej/brisque-go/internal/imageutil"
)

// testImage holds paths, description, and per-image tolerances.
// Heavily compressed and small images have wider tolerances because
// near-zero MSCN coefficients flip sign between float32 (OpenCV) and
// float64 (Go), causing systematic divergence in the AGGD fit.
var testImages = []struct {
	path     string
	desc     string
	scoreTol float64 // max acceptable |Go - OpenCV| score difference
	featTol  float64 // max acceptable per-feature difference
}{
	// --- Original images ---
	{"testdata/golden/sample.jpg", "iPhone 6 photo (EXIF rotated)", 2.0, 0.015},
	{"testdata/golden/landscape_high.jpg", "Kodak landscape, high quality", 0.1, 0.005},
	{"testdata/golden/portrait_medium.jpg", "Kodak lighthouse, medium quality", 0.2, 0.005},
	{"testdata/golden/compressed_low.jpg", "Kodak scene, heavy JPEG compression", 6.0, 0.02},
	{"testdata/golden/lena_standard.jpg", "Classic Lena, standard quality", 0.1, 0.01},
	{"testdata/golden/small_thumbnail.jpg", "Small thumbnail 192x128", 2.0, 0.06},

	// --- Quality sweep (same source, different JPEG quality) ---
	{"testdata/golden/landscape_q10.jpg", "Kodak landscape q=10", 1.0, 0.02},
	{"testdata/golden/landscape_q30.jpg", "Kodak landscape q=30", 1.0, 0.02},
	{"testdata/golden/landscape_q70.jpg", "Kodak landscape q=70", 0.1, 0.005},
	{"testdata/golden/landscape_q90.jpg", "Kodak landscape q=90", 0.1, 0.005},

	// --- Format and color space ---
	{"testdata/golden/landscape_gray.jpg", "Kodak landscape grayscale JPEG", 0.2, 0.005},
	{"testdata/golden/landscape.png", "Kodak landscape PNG (lossless)", 0.2, 0.01},

	// --- Size and aspect ratio ---
	{"testdata/golden/panoramic_wide.jpg", "Wide panoramic crop 768x128", 0.5, 0.01},
	{"testdata/golden/tiny_32x32.jpg", "Near-minimum 32x32", 2.0, 0.15},
	{"testdata/golden/large_1920.jpg", "Large upscale 1920x1280", 0.5, 0.01},

	// --- Synthetic ---
	{"testdata/golden/noise_256.jpg", "Random noise 256x256", 1.0, 0.02},
}

// TestCompareWithOpenCV runs the Python comparison script (which uses OpenCV
// without EXIF rotation to match Go's decoder) and compares features and
// scores across multiple test images.
//
// Prerequisites:
//
//	pip install opencv-contrib-python numpy
//
// Run with:
//
//	go test -tags integration -run TestCompareWithOpenCV -v
func TestCompareWithOpenCV(t *testing.T) {
	// Collect all image paths that exist
	var paths []string
	for _, ti := range testImages {
		if _, err := os.Stat(ti.path); err == nil {
			paths = append(paths, ti.path)
		}
	}
	if len(paths) == 0 {
		t.Fatal("no test images found")
	}

	// --- Python/OpenCV side (all images at once, no EXIF rotation) ---
	args := append([]string{"scripts/compare_brisque.py", "--json"}, paths...)
	cmd := exec.Command("python3", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Python script failed: %v\nOutput: %s", err, out)
	}

	type ocvResult struct {
		Image       string    `json:"image"`
		OpenCVScore *float64  `json:"opencv_score"`
		OpenCVFeats []float64 `json:"opencv_features"`
		ImageSize   *struct {
			Width  int `json:"width"`
			Height int `json:"height"`
		} `json:"image_size"`
		OpenCVError *string `json:"opencv_error"`
	}
	var ocvResults []ocvResult
	if err := json.Unmarshal(out, &ocvResults); err != nil {
		t.Fatalf("parse Python output: %v\nRaw: %s", err, out)
	}

	// Index by path
	ocvByPath := make(map[string]*ocvResult, len(ocvResults))
	for i := range ocvResults {
		ocvByPath[ocvResults[i].Image] = &ocvResults[i]
	}

	model := DefaultModel()
	ctx := context.Background()

	// --- Compare each image ---
	t.Logf("\n%-40s %10s %10s %10s %10s", "Image", "Go", "OpenCV", "ScoreDiff", "MaxFeatΔ")
	t.Logf("%-40s %10s %10s %10s %10s", "-----", "--", "------", "---------", "-------")

	for _, ti := range testImages {
		if _, err := os.Stat(ti.path); err != nil {
			continue
		}

		t.Run(filepath.Base(ti.path), func(t *testing.T) {
			// Go side
			f, err := os.Open(ti.path)
			if err != nil {
				t.Fatalf("open: %v", err)
			}
			defer f.Close()

			img, _, err := image.Decode(f)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}

			goScore, err := model.ScoreImage(ctx, img)
			if err != nil {
				t.Fatalf("ScoreImage: %v", err)
			}

			goFeats := extractGoFeatures(t, img, model)

			// OpenCV side
			ocv, ok := ocvByPath[ti.path]
			if !ok || ocv.OpenCVScore == nil || ocv.OpenCVFeats == nil {
				t.Fatalf("no OpenCV result for %s", ti.path)
			}

			ocvScore := *ocv.OpenCVScore
			scoreDiff := math.Abs(goScore - ocvScore)

			// Feature comparison
			maxFeatDiff := 0.0
			maxScale1 := 0.0
			maxScale2 := 0.0
			for i := 0; i < 36; i++ {
				d := math.Abs(goFeats[i] - ocv.OpenCVFeats[i])
				if d > maxFeatDiff {
					maxFeatDiff = d
				}
				if i < 18 && d > maxScale1 {
					maxScale1 = d
				}
				if i >= 18 && d > maxScale2 {
					maxScale2 = d
				}
			}

			t.Logf("%-40s Go=%.4f  OpenCV=%.4f  scoreDiff=%.4f  featΔ=[s1:%.6f s2:%.6f]",
				ti.desc, goScore, ocvScore, scoreDiff, maxScale1, maxScale2)

			// Log detailed feature comparison
			for i := 0; i < 36; i++ {
				d := math.Abs(goFeats[i] - ocv.OpenCVFeats[i])
				marker := ""
				if d > 0.01 {
					marker = " <<<"
				}
				t.Logf("  [%2d] Go=%-16.10f OCV=%-16.10f diff=%.10f%s",
					i, goFeats[i], ocv.OpenCVFeats[i], d, marker)
			}

			// Assertions (per-image tolerances)
			if scoreDiff > ti.scoreTol {
				t.Errorf("score diff %.4f exceeds tolerance %.1f (Go=%.4f, OpenCV=%.4f)",
					scoreDiff, ti.scoreTol, goScore, ocvScore)
			}
			if maxFeatDiff > ti.featTol {
				t.Errorf("max feature diff %.6f exceeds tolerance %.3f", maxFeatDiff, ti.featTol)
			}
		})
	}
}

// extractGoFeatures extracts the raw 36 features from the Go pipeline
// (before scaling) for comparison with OpenCV.
func extractGoFeatures(t *testing.T, img image.Image, m *Model) [36]float64 {
	t.Helper()

	ws := defaultPool.Get().(*features.Workspace)
	defer defaultPool.Put(ws)

	imageutil.FromImageInto(ws.Src, img)

	feats, err := features.Extract(context.Background(), ws.Src, m.kernel, ws)
	if err != nil {
		t.Fatalf("extractFeatures: %v", err)
	}
	return feats
}
