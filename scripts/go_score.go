//go:build ignore

package main

import (
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	brisque "github.com/matej/brisque-go"
)

func main() {
	model := brisque.DefaultModel()
	for _, path := range os.Args[1:] {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: decode error: %v\n", path, err)
			continue
		}
		score, err := model.ScoreImage(context.Background(), img)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: score error: %v\n", path, err)
			continue
		}
		fmt.Printf("%-40s Go=%.4f\n", path, score)
	}
}
