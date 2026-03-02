package imageutil

import "image"

// FloatImage is a flat float32 image representation optimized for
// numerical processing. It mirrors image.Gray's layout but uses
// float32 pixel values to match OpenCV's CV_32F precision.
type FloatImage struct {
	Pix    []float32
	Stride int
	Rect   image.Rectangle
}

// NewFloatImage allocates a new FloatImage with the given bounds.
func NewFloatImage(r image.Rectangle) *FloatImage {
	w := r.Dx()
	h := r.Dy()
	return &FloatImage{
		Pix:    make([]float32, w*h),
		Stride: w,
		Rect:   r,
	}
}

// Width returns the image width.
func (f *FloatImage) Width() int { return f.Rect.Dx() }

// Height returns the image height.
func (f *FloatImage) Height() int { return f.Rect.Dy() }

// At returns the float32 value at (x, y).
func (f *FloatImage) At(x, y int) float32 {
	if !(image.Point{X: x, Y: y}).In(f.Rect) {
		return 0
	}
	return f.Pix[(y-f.Rect.Min.Y)*f.Stride+(x-f.Rect.Min.X)]
}

// Set sets the float32 value at (x, y).
func (f *FloatImage) Set(x, y int, v float32) {
	if !(image.Point{X: x, Y: y}).In(f.Rect) {
		return
	}
	f.Pix[(y-f.Rect.Min.Y)*f.Stride+(x-f.Rect.Min.X)] = v
}

// SubImage returns a FloatImage that shares the same Pix slice but
// has its Rect restricted to r intersected with the original Rect.
func (f *FloatImage) SubImage(r image.Rectangle) *FloatImage {
	r = r.Intersect(f.Rect)
	if r.Empty() {
		return &FloatImage{Rect: r}
	}
	off := (r.Min.Y-f.Rect.Min.Y)*f.Stride + (r.Min.X - f.Rect.Min.X)
	return &FloatImage{
		Pix:    f.Pix[off:],
		Stride: f.Stride,
		Rect:   r,
	}
}

// Reset sets the rect and zeroes the pixel buffer, reusing the
// existing allocation if large enough.
func (f *FloatImage) Reset(r image.Rectangle) {
	w := r.Dx()
	h := r.Dy()
	n := w * h
	if cap(f.Pix) >= n {
		f.Pix = f.Pix[:n]
	} else {
		f.Pix = make([]float32, n)
	}
	for i := range f.Pix {
		f.Pix[i] = 0
	}
	f.Stride = w
	f.Rect = r
}

// luminance computes the ITU-R BT.601 luminance from r, g, b values
// in [0, 0xFFFF] range, returning a float32 in [0, 255] range.
func luminance(r, g, b uint32) float32 {
	return float32((0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 257.0)
}

// FromGrayBytesInto converts raw grayscale bytes into an existing FloatImage.
func FromGrayBytesInto(dst *FloatImage, pix []byte, width, height int) {
	dst.Reset(image.Rect(0, 0, width, height))
	for i, v := range pix[:width*height] {
		dst.Pix[i] = float32(v)
	}
}

// FromImageInto converts an image.Image into an existing FloatImage,
// reusing its buffer.
func FromImageInto(dst *FloatImage, img image.Image) {
	bounds := img.Bounds()
	dst.Reset(bounds)

	switch src := img.(type) {
	case *image.Gray:
		fromGray(dst, src)
	case *image.YCbCr:
		fromYCbCr(dst, src)
	case *image.RGBA:
		fromRGBA(dst, src)
	case *image.NRGBA:
		fromNRGBA(dst, src)
	default:
		fromGeneric(dst, img)
	}
}

func fromGray(dst *FloatImage, src *image.Gray) {
	bounds := src.Bounds()
	w := bounds.Dx()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		srcOff := (y-bounds.Min.Y)*src.Stride + (bounds.Min.X - src.Rect.Min.X)
		dstOff := (y - bounds.Min.Y) * dst.Stride
		srcRow := src.Pix[srcOff : srcOff+w]
		dstRow := dst.Pix[dstOff : dstOff+w]
		for x, v := range srcRow {
			dstRow[x] = float32(v)
		}
	}
}

func fromYCbCr(dst *FloatImage, src *image.YCbCr) {
	bounds := src.Bounds()
	w := bounds.Dx()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		yi := (y-bounds.Min.Y)*src.YStride + (bounds.Min.X - src.Rect.Min.X)
		dstOff := (y - bounds.Min.Y) * dst.Stride
		yRow := src.Y[yi : yi+w]
		dstRow := dst.Pix[dstOff : dstOff+w]
		for x, v := range yRow {
			dstRow[x] = float32(v)
		}
	}
}

func fromRGBA(dst *FloatImage, src *image.RGBA) {
	bounds := src.Bounds()
	w := bounds.Dx()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		srcOff := (y-bounds.Min.Y)*src.Stride + (bounds.Min.X-src.Rect.Min.X)*4
		dstOff := (y - bounds.Min.Y) * dst.Stride
		dstRow := dst.Pix[dstOff : dstOff+w]
		for x := 0; x < w; x++ {
			si := srcOff + x*4
			r := float64(src.Pix[si])
			g := float64(src.Pix[si+1])
			b := float64(src.Pix[si+2])
			dstRow[x] = float32(0.299*r + 0.587*g + 0.114*b)
		}
	}
}

func fromNRGBA(dst *FloatImage, src *image.NRGBA) {
	bounds := src.Bounds()
	w := bounds.Dx()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		srcOff := (y-bounds.Min.Y)*src.Stride + (bounds.Min.X-src.Rect.Min.X)*4
		dstOff := (y - bounds.Min.Y) * dst.Stride
		dstRow := dst.Pix[dstOff : dstOff+w]
		for x := 0; x < w; x++ {
			si := srcOff + x*4
			a := float64(src.Pix[si+3]) / 255.0
			r := float64(src.Pix[si]) * a
			g := float64(src.Pix[si+1]) * a
			b := float64(src.Pix[si+2]) * a
			dstRow[x] = float32(0.299*r + 0.587*g + 0.114*b)
		}
	}
}

func fromGeneric(dst *FloatImage, src image.Image) {
	bounds := src.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		dstOff := (y - bounds.Min.Y) * dst.Stride
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := src.At(x, y).RGBA()
			dst.Pix[dstOff+(x-bounds.Min.X)] = luminance(r, g, b)
		}
	}
}
