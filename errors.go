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
