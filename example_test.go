package brisque_test

import (
	"context"
	"fmt"
	"image"
	"image/color"

	brisque "github.com/matej/brisque-go"
)

func ExampleDefaultModel() {
	model := brisque.DefaultModel()
	fmt.Println("model loaded:", model != nil)
	// Output:
	// model loaded: true
}

func ExampleModel_ScoreImage() {
	model := brisque.DefaultModel()
	ctx := context.Background()

	// Create a simple test image
	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.SetGray(x, y, color.Gray{Y: uint8((x + y) % 256)})
		}
	}

	score, err := model.ScoreImage(ctx, img)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("score is a number: %t\n", score > -1000 && score < 1000)
	// Output:
	// score is a number: true
}

func ExampleModel_ScoreGray() {
	model := brisque.DefaultModel()
	ctx := context.Background()

	// Raw grayscale buffer (e.g., from a video decoder)
	width, height := 64, 64
	pix := make([]byte, width*height)
	for i := range pix {
		pix[i] = uint8(i % 256)
	}

	score, err := model.ScoreGray(ctx, pix, width, height)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("score is a number: %t\n", score > -1000 && score < 1000)
	// Output:
	// score is a number: true
}

func ExampleModel_ScoreBatch() {
	model := brisque.DefaultModel()
	ctx := context.Background()

	images := make([]image.Image, 3)
	for i := range images {
		img := image.NewGray(image.Rect(0, 0, 64, 64))
		for y := 0; y < 64; y++ {
			for x := 0; x < 64; x++ {
				img.SetGray(x, y, color.Gray{Y: uint8((x + y + i*30) % 256)})
			}
		}
		images[i] = img
	}

	scores, err := model.ScoreBatch(ctx, images)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("scored %d images\n", len(scores))
	// Output:
	// scored 3 images
}

func ExampleModel_ScoreWithWorkspace() {
	model := brisque.DefaultModel()
	ctx := context.Background()

	// Pre-allocate workspace for zero-allocation scoring
	ws := brisque.NewWorkspace(1920, 1080)

	img := image.NewGray(image.Rect(0, 0, 128, 128))
	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			img.SetGray(x, y, color.Gray{Y: uint8((x*3 + y*7) % 256)})
		}
	}

	score, err := model.ScoreWithWorkspace(ctx, ws, img)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("score is a number: %t\n", score > -1000 && score < 1000)
	// Output:
	// score is a number: true
}
