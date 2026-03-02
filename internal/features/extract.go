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
	tmp     []float32             // convolution temp buffer (float32 matching OpenCV)
	aggdBuf []float64             // float64 buffer for AGGD fitting conversion
	imgSq   *imageutil.FloatImage // img squared (temp)
	imgSqMu *imageutil.FloatImage // convolved img squared
	shifted *imageutil.FloatImage // shifted product temp
	half    *imageutil.FloatImage // downsampled image
}

// NewWorkspace creates a new workspace for images up to maxW × maxH.
func NewWorkspace(maxW, maxH int) *Workspace {
	maxPixels := maxW * maxH
	return &Workspace{
		Src:     imageutil.NewFloatImage(imageRect(maxW, maxH)),
		mu:      imageutil.NewFloatImage(imageRect(maxW, maxH)),
		sigma:   imageutil.NewFloatImage(imageRect(maxW, maxH)),
		mscn:    imageutil.NewFloatImage(imageRect(maxW, maxH)),
		tmp:     make([]float32, maxPixels),
		aggdBuf: make([]float64, maxPixels),
		imgSq:   imageutil.NewFloatImage(imageRect(maxW, maxH)),
		imgSqMu: imageutil.NewFloatImage(imageRect(maxW, maxH)),
		shifted: imageutil.NewFloatImage(imageRect(maxW, maxH)),
		half:    imageutil.NewFloatImage(imageRect(maxW/2, maxH/2)),
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

	// Downsample original for scale 2 using bicubic interpolation
	// matching OpenCV's cv::resize with INTER_CUBIC.
	ResizeCubicHalf(ws.half, img)

	// Scale 2: half
	f2, err := extractScale(ws.half, kernel, ws)
	if err != nil {
		return features, err
	}
	copy(features[18:], f2[:])

	return features, nil
}

// extractScale extracts 18 features from a single scale, matching the
// OpenCV BRISQUE implementation (qualitybrisque.cpp).
func extractScale(img *imageutil.FloatImage, kernel []float64, ws *Workspace) ([18]float64, error) {
	var features [18]float64
	w := img.Width()
	h := img.Height()
	n := w * h

	if w < len(kernel) || h < len(kernel) {
		return features, &tooSmallError{w, h, len(kernel)}
	}

	// 1. mu = GaussianBlur(I, BORDER_REPLICATE) — same size as input
	conv.ConvolveReplicate(ws.mu, img, kernel, ws.tmp)

	// 2. sigma = sqrt(max(0, GaussianBlur(I^2) - mu^2)) + C
	ws.imgSq.Reset(img.Rect)
	for i := 0; i < n; i++ {
		ws.imgSq.Pix[i] = img.Pix[i] * img.Pix[i]
	}
	conv.ConvolveReplicate(ws.imgSqMu, ws.imgSq, kernel, ws.tmp)

	ws.mscn.Reset(img.Rect)
	ws.sigma.Reset(img.Rect)
	for i := 0; i < n; i++ {
		muVal := ws.mu.Pix[i]
		varVal := ws.imgSqMu.Pix[i] - muVal*muVal
		if varVal < 0 {
			varVal = 0
		}
		// C = 1.0 on [0,255] range (matches OpenCV)
		// math.Sqrt is float64 but result is stored as float32
		sigVal := float32(math.Sqrt(float64(varVal))) + 1.0
		ws.sigma.Pix[i] = sigVal
		ws.mscn.Pix[i] = (img.Pix[i] - muVal) / sigVal
	}

	// 3. Fit AGGD to MSCN coefficients (matching OpenCV which converts
	//    float MSCN to vector<double> before calling AGGDfit)
	f64Buf := ws.aggdBuf[:n]
	for i := 0; i < n; i++ {
		f64Buf[i] = float64(ws.mscn.Pix[i])
	}
	aggdAlpha, ls2, rs2, _, err := stats.FitAGGD(f64Buf)
	if err != nil {
		return features, err
	}
	features[0] = aggdAlpha
	features[1] = (ls2 + rs2) / 2.0 // OpenCV: (lsigma^2 + rsigma^2) / 2

	// 4. Pairwise shifted products with zero-padding (matching OpenCV).
	// OpenCV creates a full-size shifted image, zeros where shift goes OOB,
	// then multiplies. The zeros are included in the AGGD fit.
	type shift struct {
		dy, dx int
	}
	// OpenCV shifts: {{0,1},{1,0},{1,1},{-1,1}} where [0]=row, [1]=col
	shifts := [4]shift{
		{0, 1},  // H: right neighbor
		{1, 0},  // V: below neighbor
		{1, 1},  // D1: below-right
		{-1, 1}, // D2: above-right
	}

	for si, s := range shifts {
		pairData := ws.shifted.Pix[:n]
		// Zero-initialize (zeros match OpenCV's out-of-bounds padding)
		for i := range pairData {
			pairData[i] = 0
		}
		// Fill valid products (float32 arithmetic matching OpenCV)
		for y := 0; y < h; y++ {
			sy := y + s.dy
			if sy < 0 || sy >= h {
				continue
			}
			srcRow := y * w
			shiftRow := sy * w
			for x := 0; x < w; x++ {
				sx := x + s.dx
				if sx < 0 || sx >= w {
					continue
				}
				pairData[srcRow+x] = ws.mscn.Pix[srcRow+x] * ws.mscn.Pix[shiftRow+sx]
			}
		}

		// Convert float32 pair data to float64 for AGGD fitting
		// (matches OpenCV's vector<double> conversion)
		pairF64 := ws.aggdBuf[:n]
		for i := 0; i < n; i++ {
			pairF64[i] = float64(pairData[i])
		}

		pAlpha, pLeftS2, pRightS2, pMean, err := stats.FitAGGD(pairF64)
		if err != nil {
			return features, err
		}

		base := 2 + si*4
		features[base] = pAlpha
		features[base+1] = pMean
		features[base+2] = pLeftS2
		features[base+3] = pRightS2
	}

	return features, nil
}

func imageRect(w, h int) image.Rectangle {
	return image.Rect(0, 0, w, h)
}

type tooSmallError struct {
	w, h, minDim int
}

func (e *tooSmallError) Error() string {
	return "image too small for convolution"
}
