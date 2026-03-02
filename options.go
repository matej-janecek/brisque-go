package brisque

import "sync"

// Logger is an optional logger interface.
type Logger interface {
	Printf(format string, v ...interface{})
}

// Option configures a Model.
type Option func(*config)

type config struct {
	workspacePool     *sync.Pool
	parallelThreshold int
	logger            Logger
	unsafeOpts        bool
}

func defaultConfig() config {
	return config{
		parallelThreshold: 1_000_000, // 1 MP
	}
}

// WithWorkspacePool sets a sync.Pool for workspace reuse. If not set,
// an internal pool is used.
func WithWorkspacePool(p *sync.Pool) Option {
	return func(c *config) {
		c.workspacePool = p
	}
}

// WithParallelThreshold sets the pixel count above which the two BRISQUE
// scales are computed in parallel. Default is 1,000,000 (1 MP).
// Set to 0 to always run in parallel, or math.MaxInt to never parallelize.
func WithParallelThreshold(pixels int) Option {
	return func(c *config) {
		c.parallelThreshold = pixels
	}
}

// WithLogger sets a logger for debug output.
func WithLogger(l Logger) Option {
	return func(c *config) {
		c.logger = l
	}
}

// WithUnsafeOptimizations enables unsafe memory optimizations such as
// zero-copy image conversion. Only use when you understand the implications.
func WithUnsafeOptimizations() Option {
	return func(c *config) {
		c.unsafeOpts = true
	}
}
