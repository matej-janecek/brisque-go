package brisque

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/matej/brisque-go/internal/conv"
	"github.com/matej/brisque-go/internal/svr"
)

// Model holds the BRISQUE model parameters. It is immutable and safe for
// concurrent use after construction.
type Model struct {
	// Gaussian kernel parameters
	kernelSigma float64
	kernelSize  int
	kernel      []float64

	// SVR model
	svr *svr.Model

	// Feature scaling ranges
	scaleMins [36]float64
	scaleMaxs [36]float64

	// Runtime config
	cfg config
}

// NewModel creates a Model from a reader containing the model data in
// the library's binary format. Use LoadModelFromFile for convenience.
func NewModel(r io.Reader, opts ...Option) (*Model, error) {
	m := &Model{
		cfg: defaultConfig(),
	}
	for _, o := range opts {
		o(&m.cfg)
	}

	if err := m.parse(r); err != nil {
		return nil, fmt.Errorf("brisque: failed to parse model: %w", err)
	}

	m.kernel = conv.MakeGaussianKernel(m.kernelSigma, m.kernelSize)
	return m, nil
}

// DefaultModel returns a Model using the embedded default OpenCV BRISQUE
// model. This requires no external files.
func DefaultModel(opts ...Option) *Model {
	m := &Model{
		cfg: defaultConfig(),
	}
	for _, o := range opts {
		o(&m.cfg)
	}

	m.kernelSigma = 7.0 / 6.0
	m.kernelSize = 7
	m.kernel = conv.MakeGaussianKernel(m.kernelSigma, m.kernelSize)

	m.svr = defaultSVRModel()
	m.scaleMins = defaultScaleMins
	m.scaleMaxs = defaultScaleMaxs

	return m
}

// parse reads the model from a text format:
// Line 1: kernel_sigma kernel_size
// Line 2: gamma rho nsv
// Line 3..3+nsv: alpha sv[0] sv[1] ... sv[35]
// Line 3+nsv+1: min[0] min[1] ... min[35]
// Line 3+nsv+2: max[0] max[1] ... max[35]
func (m *Model) parse(r io.Reader) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Line 1: kernel params
	if !scanner.Scan() {
		return fmt.Errorf("unexpected end of input")
	}
	parts := strings.Fields(scanner.Text())
	if len(parts) != 2 {
		return fmt.Errorf("expected 2 kernel params, got %d", len(parts))
	}
	var err error
	m.kernelSigma, err = strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return fmt.Errorf("kernel sigma: %w", err)
	}
	m.kernelSize, err = strconv.Atoi(parts[1])
	if err != nil {
		return fmt.Errorf("kernel size: %w", err)
	}

	// Line 2: SVR params
	if !scanner.Scan() {
		return fmt.Errorf("unexpected end of input")
	}
	parts = strings.Fields(scanner.Text())
	if len(parts) != 3 {
		return fmt.Errorf("expected 3 SVR params, got %d", len(parts))
	}
	gamma, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return fmt.Errorf("gamma: %w", err)
	}
	rho, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return fmt.Errorf("rho: %w", err)
	}
	nsv, err := strconv.Atoi(parts[2])
	if err != nil {
		return fmt.Errorf("nsv: %w", err)
	}

	// Lines 3..3+nsv: support vectors
	alphas := make([]float64, nsv)
	svecs := make([]float64, nsv*36)
	for i := 0; i < nsv; i++ {
		if !scanner.Scan() {
			return fmt.Errorf("unexpected end at SV %d", i)
		}
		parts = strings.Fields(scanner.Text())
		if len(parts) != 37 {
			return fmt.Errorf("SV %d: expected 37 fields, got %d", i, len(parts))
		}
		alphas[i], err = strconv.ParseFloat(parts[0], 64)
		if err != nil {
			return fmt.Errorf("SV %d alpha: %w", i, err)
		}
		for j := 0; j < 36; j++ {
			svecs[i*36+j], err = strconv.ParseFloat(parts[1+j], 64)
			if err != nil {
				return fmt.Errorf("SV %d feature %d: %w", i, j, err)
			}
		}
	}

	m.svr = &svr.Model{
		SupportVectors: svecs,
		Alpha:          alphas,
		Rho:            rho,
		Gamma:          gamma,
		NSV:            nsv,
	}

	// Scale ranges
	if !scanner.Scan() {
		return fmt.Errorf("unexpected end at scale mins")
	}
	parts = strings.Fields(scanner.Text())
	if len(parts) != 36 {
		return fmt.Errorf("expected 36 scale mins, got %d", len(parts))
	}
	for i := 0; i < 36; i++ {
		m.scaleMins[i], err = strconv.ParseFloat(parts[i], 64)
		if err != nil {
			return fmt.Errorf("scale min %d: %w", i, err)
		}
	}

	if !scanner.Scan() {
		return fmt.Errorf("unexpected end at scale maxs")
	}
	parts = strings.Fields(scanner.Text())
	if len(parts) != 36 {
		return fmt.Errorf("expected 36 scale maxs, got %d", len(parts))
	}
	for i := 0; i < 36; i++ {
		m.scaleMaxs[i], err = strconv.ParseFloat(parts[i], 64)
		if err != nil {
			return fmt.Errorf("scale max %d: %w", i, err)
		}
	}

	return scanner.Err()
}

// LoadModelFromFile is a convenience function that loads a model from a file path.
func LoadModelFromFile(path string, opts ...Option) (m *Model, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("brisque: %w", err)
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("brisque: %w", cerr)
		}
	}()
	return NewModel(f, opts...)
}
