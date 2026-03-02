package brisque

import "fmt"

// ErrImageTooSmall reports that the input image dimensions are below
// the minimum required for BRISQUE feature extraction.
type ErrImageTooSmall struct {
	Width, Height       int
	MinWidth, MinHeight int
}

func (e *ErrImageTooSmall) Error() string {
	return fmt.Sprintf("brisque: image too small (%dx%d), minimum required is %dx%d",
		e.Width, e.Height, e.MinWidth, e.MinHeight)
}

// ErrInvalidFeatureCount reports a mismatch between the expected and
// actual number of features.
type ErrInvalidFeatureCount struct {
	Expected int
	Actual   int
}

func (e *ErrInvalidFeatureCount) Error() string {
	return fmt.Sprintf("brisque: invalid feature count: expected %d, got %d",
		e.Expected, e.Actual)
}

// ErrUniformImage indicates that the image has zero variance and
// cannot be scored meaningfully.
type ErrUniformImage struct{}

func (e *ErrUniformImage) Error() string {
	return "brisque: uniform image (zero variance), cannot compute quality score"
}

// ErrDegenerateDistribution indicates that the GGD or AGGD fitting
// failed due to degenerate input data.
type ErrDegenerateDistribution struct {
	Scale   int
	Feature string
}

func (e *ErrDegenerateDistribution) Error() string {
	return fmt.Sprintf("brisque: degenerate distribution at scale %d, feature %q",
		e.Scale, e.Feature)
}
