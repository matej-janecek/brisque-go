//go:build ignore

package main

import (
	"encoding/binary"
	"image"
	_ "image/jpeg"
	"math"
	"os"

	"github.com/matej-janecek/brisque-go/internal/imageutil"
)

func main() {
	f, _ := os.Open(os.Args[1])
	defer f.Close()
	img, _, _ := image.Decode(f)

	fi := imageutil.NewFloatImage(img.Bounds())
	imageutil.FromImageInto(fi, img)

	w, h := fi.Width(), fi.Height()
	binary.Write(os.Stdout, binary.LittleEndian, uint32(w))
	binary.Write(os.Stdout, binary.LittleEndian, uint32(h))
	for i := 0; i < w*h; i++ {
		binary.Write(os.Stdout, binary.LittleEndian, math.Float32bits(fi.Pix[i]))
	}
}
