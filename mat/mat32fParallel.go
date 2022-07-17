package mat

import (
	"log"
	"runtime"
	"sync"
)

func MatMulParallel(a, b *M32f) *M32f {
	if a.shape.D[0] < 1000 && a.shape.D[1] < 1000 {
		return MatMul(a, b)
	}

	if a.shape.D[0] == 0 || a.shape.D[1] == 0 || b.shape.D[0] == 0 || b.shape.D[1] == 0 || a.shape.D[1] != b.shape.D[0] {
		log.Fatalf("cannot multiply matrices of incompatible sizes -> %dx%d and %dx%d", a.shape.D[0], a.shape.D[1], b.shape.D[0], b.shape.D[1])
	}

	wg := &sync.WaitGroup{}
	wg.Add(a.shape.D[0])
	in := make(chan int, a.shape.D[0])
	result := make([]float32, a.shape.D[0]*b.shape.D[1])
	for i := 0; i < runtime.NumCPU(); i++ {
		go matMulParallelChunk(result, a, b, in, wg)
	}

	for i := 0; i < a.shape.D[0]; i++ {
		in <- i
	}

	wg.Wait()
	close(in)
	return FromSlice32f(Dim(a.shape.D[0], b.shape.D[1]), result)
}

func matMulParallelChunk(result []float32, a, b *M32f, in chan int, wg *sync.WaitGroup) {
	for i := range in {
		for j := 0; j < b.shape.D[1]; j++ {
			for k := 0; k < b.shape.D[0]; k++ {
				result[i*b.shape.D[1]+j] += a.data[i*b.shape.D[0]+k] * b.data[k*b.shape.D[1]+j]
			}
		}

		wg.Done()
	}
}
