package brisque

import (
	"github.com/matej-janecek/brisque-go/internal/features"
)

// Workspace holds pre-allocated buffers for BRISQUE computation.
// Create one with NewWorkspace and reuse across calls to
// ScoreWithWorkspace for zero-allocation scoring.
//
// A Workspace is NOT safe for concurrent use. Each goroutine should
// have its own Workspace, or use a sync.Pool.
type Workspace struct {
	fw *features.Workspace
}

// NewWorkspace creates a Workspace pre-allocated for images up to
// maxWidth × maxHeight pixels.
func NewWorkspace(maxWidth, maxHeight int) *Workspace {
	return &Workspace{
		fw: features.NewWorkspace(maxWidth, maxHeight),
	}
}
