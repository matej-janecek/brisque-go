//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	"os"

	"github.com/matej/brisque-go/internal/imageutil"
)

func main() {
	f, _ := os.Open(os.Args[1])
	defer f.Close()
	img, _, _ := image.Decode(f)

	fi := imageutil.NewFloatImage(img.Bounds())
	imageutil.FromImageInto(fi, img)

	w, h := fi.Width(), fi.Height()
	var sum, sumSq float64
	minV, maxV := float64(fi.Pix[0]), float64(fi.Pix[0])
	for i := 0; i < w*h; i++ {
		v := float64(fi.Pix[i])
		sum += v
		sumSq += v * v
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	mean := sum / float64(w*h)
	std := (sumSq/float64(w*h) - mean*mean)

	block := make([][]float32, 20)
	for y := 0; y < 20; y++ {
		block[y] = make([]float32, 20)
		for x := 0; x < 20; x++ {
			block[y][x] = fi.Pix[y*fi.Stride+x]
		}
	}

	out, _ := json.Marshal(map[string]interface{}{
		"width":          w,
		"height":         h,
		"mean":           mean,
		"variance":       std,
		"min":            minV,
		"max":            maxV,
		"top_left_20x20": block,
	})
	fmt.Println(string(out))
}
