package features

import (
	"image"
	"math"

	"github.com/matej/brisque-go/internal/conv"
	"github.com/matej/brisque-go/internal/imageutil"
	"github.com/matej/brisque-go/internal/stats"
)

// NumFeatures is the total number of BRISQUE features (18 per scale × 2 scales).
const NumFeatures = 36

// Workspace holds pre-allocated buffers for feature extraction.
type Workspace struct {
	Src     *imageutil.FloatImage // source image buffer (for zero-alloc path)
	mu      *imageutil.FloatImage // local mean
	sigma   *imageutil.FloatImage // local variance -> std dev
	mscn    *imageutil.FloatImage // MSCN coefficients
	tmp     []float64             // convolution temp buffer
	muSq    *imageutil.FloatImage // mu squared (temp)
	imgSq   *imageutil.FloatImage // img squared (temp)
	imgSqMu *imageutil.FloatImage // convolved img squared
	shifted *imageutil.FloatImage // shifted product temp
	half    *imageutil.FloatImage // downsampled image
}

// NewWorkspace creates a new workspace for images up to maxW × maxH.
func NewWorkspace(maxW, maxH int) *Workspace {
	maxPixels := maxW * maxH
	return &Workspace{
		Src:     imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		mu:      imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		sigma:   imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		mscn:    imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		tmp:     make([]float64, maxPixels),
		muSq:    imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		imgSq:   imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		imgSqMu: imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		shifted: imageutil.NewFloatImage(image.Rect(0, 0, maxW, maxH)),
		half:    imageutil.NewFloatImage(image.Rect(0, 0, maxW/2, maxH/2)),
	}
}

// Extract computes all 36 BRISQUE features from the given float image.
// kernel is the precomputed 1D Gaussian kernel. ws is a pre-allocated workspace.
func Extract(img *imageutil.FloatImage, kernel []float64, ws *Workspace) ([NumFeatures]float64, error) {
	var features [NumFeatures]float64

	// Scale 1: original
	f1, err := extractScale(img, kernel, ws)
	if err != nil {
		return features, err
	}
	copy(features[:18], f1[:])

	// Downsample for scale 2
	Downsample2x(ws.half, img)

	// Scale 2: half
	f2, err := extractScale(ws.half, kernel, ws)
	if err != nil {
		return features, err
	}
	copy(features[18:], f2[:])

	return features, nil
}

// extractScale extracts 18 features from a single scale.
func extractScale(img *imageutil.FloatImage, kernel []float64, ws *Workspace) ([18]float64, error) {
	var features [18]float64
	w := img.Width()
	h := img.Height()
	ksize := len(kernel)

	if w < ksize || h < ksize {
		return features, &tooSmallError{w, h, ksize}
	}

	// 1. Compute local mean: mu = G * I
	conv.Convolve(ws.mu, img, kernel, ws.tmp)

	muW := ws.mu.Width()
	muH := ws.mu.Height()
	n := muW * muH

	// 2. Compute I^2 into imgSq (within the valid region of mu)
	ws.imgSq.Reset(ws.mu.Rect)
	half := ksize / 2
	for y := 0; y < muH; y++ {
		imgRow := img.Pix[(y+half)*img.Stride+half : (y+half)*img.Stride+half+muW]
		sqRow := ws.imgSq.Pix[y*ws.imgSq.Stride : y*ws.imgSq.Stride+muW]
		for x := 0; x < muW; x++ {
			sqRow[x] = imgRow[x] * imgRow[x]
		}
	}

	// 3. Compute G * I^2
	conv.Convolve(ws.imgSqMu, ws.imgSq, kernel, ws.tmp)

	// The valid region of imgSqMu is smaller than mu's region.
	// We need to work in the intersection.
	sigW := ws.imgSqMu.Width()
	sigH := ws.imgSqMu.Height()

	// Actually, let's take a simpler approach that matches the standard BRISQUE:
	// Compute mu and mu^2 in the same valid region.
	// sigma = sqrt(abs(G * I^2 - mu^2)) + C

	// Recompute: use mu's region for MSCN computation directly
	// Simpler approach: compute MSCN in mu's valid region
	ws.mscn.Reset(ws.mu.Rect)
	ws.sigma.Reset(ws.mu.Rect)

	// Compute sigma = sqrt(|conv(I^2) - mu^2|)
	// But conv(I^2) has a smaller valid region.
	// Standard approach: compute mu and sigma using the same convolution bounds.

	// Let's use the standard approach:
	// mu = conv(I, G)
	// sigma = sqrt(conv(I^2, G) - mu^2)
	// Both conv operations produce same-size output from same-size input
	// So we convolve the FULL image for both.

	// Reconvolve I^2 from scratch on the full image
	ws.imgSq.Reset(img.Rect)
	for i, v := range img.Pix[:w*h] {
		ws.imgSq.Pix[i] = v * v
	}
	conv.Convolve(ws.imgSqMu, ws.imgSq, kernel, ws.tmp)

	// Now mu and imgSqMu have the same dimensions
	sigW = ws.imgSqMu.Width()
	sigH = ws.imgSqMu.Height()
	_ = sigW
	_ = sigH

	// Compute MSCN coefficients
	ws.mscn.Reset(ws.mu.Rect)
	ws.sigma.Reset(ws.mu.Rect)
	for i := 0; i < n; i++ {
		muVal := ws.mu.Pix[i]
		varVal := ws.imgSqMu.Pix[i] - muVal*muVal
		if varVal < 0 {
			varVal = 0
		}
		sigVal := math.Sqrt(varVal) + 1.0 // C = 1 stabilization constant
		ws.sigma.Pix[i] = sigVal
		// MSCN = (I - mu) / (sigma + C)
		// Get the corresponding pixel from original image
		iy := i/ws.mu.Stride + half
		ix := i%ws.mu.Stride + half
		origVal := img.Pix[iy*img.Stride+ix]
		ws.mscn.Pix[i] = (origVal - muVal) / sigVal
	}

	// 3. Fit GGD to MSCN coefficients
	mscnData := ws.mscn.Pix[:n]
	alpha, sigma2, err := stats.FitGGD(mscnData)
	if err != nil {
		return features, err
	}
	features[0] = alpha
	features[1] = sigma2

	// 4. Compute 4 pairwise products and fit AGGD
	mscnW := ws.mscn.Width()
	mscnH := ws.mscn.Height()

	type shift struct {
		dx, dy int
		name   string
	}
	shifts := [4]shift{
		{1, 0, "H"},   // horizontal
		{0, 1, "V"},   // vertical
		{1, 1, "D1"},  // diagonal 1
		{1, -1, "D2"}, // diagonal 2
	}

	for si, s := range shifts {
		// Compute pairwise product between MSCN(x,y) and MSCN(x+dx, y+dy)
		// Valid region shrinks by the shift amount
		pW := mscnW - abs(s.dx)
		pH := mscnH - abs(s.dy)
		if pW <= 0 || pH <= 0 {
			continue
		}

		startY := 0
		if s.dy < 0 {
			startY = -s.dy
		}
		endY := startY + pH

		pairData := ws.shifted.Pix[:pW*pH]
		idx := 0
		for y := startY; y < endY; y++ {
			for x := 0; x < pW; x++ {
				v1 := ws.mscn.Pix[y*mscnW+x]
				v2 := ws.mscn.Pix[(y+s.dy)*mscnW+(x+s.dx)]
				pairData[idx] = v1 * v2
				idx++
			}
		}

		aggdAlpha, leftSigma2, rightSigma2, mean, err := stats.FitAGGD(pairData[:idx])
		if err != nil {
			return features, err
		}

		base := 2 + si*4
		features[base] = aggdAlpha
		features[base+1] = mean
		features[base+2] = leftSigma2
		features[base+3] = rightSigma2

		_ = s.name
	}

	return features, nil
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

type tooSmallError struct {
	w, h, minDim int
}

func (e *tooSmallError) Error() string {
	return "image too small for convolution"
}
