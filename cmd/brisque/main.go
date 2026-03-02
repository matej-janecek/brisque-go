package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"

	brisque "github.com/matej/brisque-go"
)

type result struct {
	File  string  `json:"file"`
	Score float64 `json:"score"`
	Error string  `json:"error,omitempty"`
}

func main() {
	jsonOutput := flag.Bool("json", false, "output results as JSON")
	workers := flag.Int("workers", runtime.GOMAXPROCS(0), "number of concurrent workers")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		_, _ = fmt.Fprintln(os.Stderr, "usage: brisque [flags] <image> [image...]")
		_, _ = fmt.Fprintln(os.Stderr, "       brisque [flags] <directory>")
		_, _ = fmt.Fprintln(os.Stderr, "       brisque [flags] <glob-pattern>")
		_, _ = fmt.Fprintln(os.Stderr, "\nflags:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Expand arguments: directories, globs
	files, err := expandArgs(args)
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	if len(files) == 0 {
		_, _ = fmt.Fprintln(os.Stderr, "error: no image files found")
		os.Exit(1)
	}

	// Setup context with signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		cancel()
	}()

	model := brisque.DefaultModel()
	results := processFiles(ctx, model, files, *workers)

	hasError := false
	if *jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(results); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "error encoding JSON: %v\n", err)
			hasError = true
		}
	} else {
		for _, r := range results {
			if r.Error != "" {
				_, _ = fmt.Fprintf(os.Stderr, "%s: error: %s\n", r.File, r.Error)
				hasError = true
			} else {
				fmt.Printf("%s\t%.4f\n", r.File, r.Score)
			}
		}
	}

	if hasError {
		os.Exit(1)
	}
}

func expandArgs(args []string) ([]string, error) {
	var files []string
	for _, arg := range args {
		info, err := os.Stat(arg)
		if err == nil && info.IsDir() {
			// Directory: find image files
			entries, err := os.ReadDir(arg)
			if err != nil {
				return nil, fmt.Errorf("reading directory %s: %w", arg, err)
			}
			for _, e := range entries {
				if !e.IsDir() && isImageFile(e.Name()) {
					files = append(files, filepath.Join(arg, e.Name()))
				}
			}
			continue
		}

		// Try glob
		matches, err := filepath.Glob(arg)
		if err != nil {
			return nil, fmt.Errorf("invalid glob pattern %s: %w", arg, err)
		}
		if len(matches) > 0 {
			for _, m := range matches {
				info, err := os.Stat(m)
				if err == nil && !info.IsDir() && isImageFile(m) {
					files = append(files, m)
				}
			}
			continue
		}

		// Treat as a single file
		files = append(files, arg)
	}
	return files, nil
}

func isImageFile(name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif":
		return true
	}
	return false
}

func processFiles(ctx context.Context, model *brisque.Model, files []string, workers int) []result {
	if workers < 1 {
		workers = 1
	}
	if workers > len(files) {
		workers = len(files)
	}

	results := make([]result, len(files))
	jobs := make(chan int, len(files))

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				select {
				case <-ctx.Done():
					results[idx] = result{File: files[idx], Error: "cancelled"}
					return
				default:
				}
				score, err := scoreFile(ctx, model, files[idx])
				if err != nil {
					results[idx] = result{File: files[idx], Error: err.Error()}
				} else {
					results[idx] = result{File: files[idx], Score: score}
				}
			}
		}()
	}

	for i := range files {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	return results
}

func scoreFile(ctx context.Context, model *brisque.Model, path string) (score float64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("closing %s: %w", path, cerr)
		}
	}()

	img, _, err := image.Decode(f)
	if err != nil {
		return 0, fmt.Errorf("decoding image: %w", err)
	}

	return model.ScoreImage(ctx, img)
}
