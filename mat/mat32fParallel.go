package mat

import (
	"log"
	"runtime"
	"sync"
)

func MatMulParallel(a, b *Mat32f) *Mat32f {
	if a.shape.X == 0 || a.shape.Y == 0 || b.shape.X == 0 || b.shape.Y == 0 || a.shape.Y != b.shape.X {
		log.Fatalf("cannot multiply matrices of incompatible sizes -> %dx%d and %dx%d", a.shape.X, a.shape.Y, b.shape.X, b.shape.Y)
	}

	wg := &sync.WaitGroup{}
	wg.Add(a.shape.X)
	in := make(chan int, a.shape.X)
	result := make([]float32, a.shape.X * b.shape.Y)
	for i := 0; i < runtime.NumCPU(); i++ {
		go matMulParallelChunk(result, a, b, in, wg)
	}

	for i := 0; i < a.shape.X; i++ {
		in <- i
	}

	wg.Wait()
	close(in)
	return NewMat32f(WithShape(a.shape.X, b.shape.Y), result)
}

func matMulParallelChunk(result []float32, a, b *Mat32f, in chan int, wg *sync.WaitGroup) {
	for i := range in {
		for j := 0; j < b.shape.Y; j++ {
			for k := 0; k < b.shape.X; k++ {
				result[i * b.shape.Y + j] += a.data[i * b.shape.X + k] * b.data[k * b.shape.Y + j]
			}
		}

		wg.Done()
	}
}
