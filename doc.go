// Package brisque implements the BRISQUE (Blind/Referenceless Image Spatial
// Quality Evaluator) algorithm for no-reference image quality assessment.
//
// BRISQUE scores range from 0 to 100, where lower scores indicate better
// perceptual quality. Typical scores for natural images fall in the 0-60
// range; heavily compressed or corrupted images score 60-100.
//
// # Quick Start
//
// Create a model with the embedded default (trained on the LIVE-R2 database),
// then score any image.Image:
//
//	model := brisque.DefaultModel()
//	score, err := model.ScoreImage(ctx, img)
//
// For raw grayscale bytes (e.g., from a video decoder or custom JPEG decoder):
//
//	score, err := model.ScoreGray(ctx, pixelBytes, width, height)
//
// For zero-allocation hot paths, pre-allocate a [Workspace]:
//
//	ws := brisque.NewWorkspace(1920, 1080)
//	score, err := model.ScoreWithWorkspace(ctx, ws, img)
//
// # Concurrency
//
// A [Model] is immutable and safe for concurrent use from multiple goroutines.
// A [Workspace] is NOT safe for concurrent use; each goroutine needs its own
// instance (or use a sync.Pool).
// [ScoreBatch] processes multiple images concurrently using a bounded worker pool.
//
// # Image Requirements
//
// Images must be at least 16x16 pixels. Smaller images return [ErrImageTooSmall].
// All standard image types are supported (JPEG, PNG, GIF, etc.); JPEG images
// use a fast path that extracts the Y (luminance) channel directly.
//
// # Error Handling
//
// All scoring methods return typed errors that can be inspected with type assertions:
//
//   - [ErrImageTooSmall]: image dimensions below 16x16
//   - [ErrUniformImage]: solid-color image with zero variance
//   - [ErrDegenerateDistribution]: statistical fitting failure (rare, usually synthetic images)
package brisque
