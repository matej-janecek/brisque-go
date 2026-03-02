package brisque

import (
	"context"
	"image"
	"runtime"
	"sync"

	"github.com/matej/brisque-go/internal/features"
	"github.com/matej/brisque-go/internal/imageutil"
	"github.com/matej/brisque-go/internal/scaler"
	"github.com/matej/brisque-go/internal/svr"
)

// minImageDim is the minimum dimension required for BRISQUE.
// After downsampling 2x and applying a 7-wide kernel, we need at least
// kernel_size pixels in each dimension at the half scale.
const minImageDim = 16

var defaultPool = &sync.Pool{
	New: func() interface{} {
		return features.NewWorkspace(3840, 2160)
	},
}

// ScoreImage computes the BRISQUE quality score for the given image.
// Lower scores indicate better perceptual quality (typical range 0–100).
func (m *Model) ScoreImage(ctx context.Context, img image.Image) (float64, error) {
	pool := m.cfg.workspacePool
	if pool == nil {
		pool = defaultPool
	}
	ws := pool.Get().(*features.Workspace)
	defer pool.Put(ws)

	return m.scoreWithWorkspace(ctx, ws, img)
}

// ScoreGray computes the BRISQUE score from raw grayscale pixel data.
// pix must contain width*height bytes in row-major order.
func (m *Model) ScoreGray(ctx context.Context, pix []byte, width, height int) (float64, error) {
	if width < minImageDim || height < minImageDim {
		return 0, &ErrImageTooSmall{
			Width: width, Height: height,
			MinWidth: minImageDim, MinHeight: minImageDim,
		}
	}

	pool := m.cfg.workspacePool
	if pool == nil {
		pool = defaultPool
	}
	ws := pool.Get().(*features.Workspace)
	defer pool.Put(ws)

	imageutil.FromGrayBytesInto(ws.Src, pix, width, height)
	return m.predict(ctx, ws, ws.Src)
}

// ScoreWithWorkspace computes the BRISQUE score using a pre-allocated
// workspace for zero-allocation operation.
func (m *Model) ScoreWithWorkspace(ctx context.Context, ws *Workspace, img image.Image) (float64, error) {
	return m.scoreWithWorkspace(ctx, ws.fw, img)
}

// ScoreBatch computes BRISQUE scores for multiple images concurrently.
// Returns scores in the same order as the input slice.
// Processing stops on the first error or context cancellation.
func (m *Model) ScoreBatch(ctx context.Context, images []image.Image) ([]float64, error) {
	n := len(images)
	if n == 0 {
		return nil, nil
	}

	scores := make([]float64, n)
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}

	pool := m.cfg.workspacePool
	if pool == nil {
		pool = defaultPool
	}

	type result struct {
		idx   int
		score float64
		err   error
	}

	jobs := make(chan int, n)
	results := make(chan result, n)

	// Start workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ws := pool.Get().(*features.Workspace)
			defer pool.Put(ws)

			for idx := range jobs {
				select {
				case <-ctx.Done():
					results <- result{idx: idx, err: ctx.Err()}
					return
				default:
				}
				score, err := m.scoreWithWorkspace(ctx, ws, images[idx])
				results <- result{idx: idx, score: score, err: err}
			}
		}()
	}

	// Send jobs
	go func() {
		for i := 0; i < n; i++ {
			select {
			case <-ctx.Done():
				close(jobs)
				return
			case jobs <- i:
			}
		}
		close(jobs)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		if r.err != nil {
			return nil, r.err
		}
		scores[r.idx] = r.score
	}

	return scores, nil
}

func (m *Model) scoreWithWorkspace(ctx context.Context, ws *features.Workspace, img image.Image) (float64, error) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w < minImageDim || h < minImageDim {
		return 0, &ErrImageTooSmall{
			Width: w, Height: h,
			MinWidth: minImageDim, MinHeight: minImageDim,
		}
	}

	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}

	// Use workspace's pre-allocated source buffer (zero-alloc path)
	imageutil.FromImageInto(ws.Src, img)
	return m.predict(ctx, ws, ws.Src)
}

func (m *Model) predict(ctx context.Context, ws *features.Workspace, fi *imageutil.FloatImage) (float64, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	default:
	}

	feats, err := features.Extract(fi, m.kernel, ws)
	if err != nil {
		return 0, err
	}

	scaler.Scale(&feats, m.scaleMins, m.scaleMaxs)
	score := svr.Predict(feats, m.svr)

	// Clamp to [0, 100] matching OpenCV's qualitybrisque.cpp
	if score < 0 {
		score = 0
	} else if score > 100 {
		score = 100
	}

	return score, nil
}
