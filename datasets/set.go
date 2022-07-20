package datasets

import (
	"math/rand"
	"ml-framework/mat"
)

const TrainingSetX = "trainingSetX"
const TrainingSetY = "trainingSetY"
const ValidationSetX = "validationSetX"
const ValidationSetY = "validationSetY"

type set struct {
	data          *mat.M32f
	batchSize     int
	batchCounter  int
	shouldShuffle bool
}

func newSet(batchSize int) *set {
	return &set{
		batchSize: batchSize,
	}
}

func (s *set) SetData(mat *mat.M32f) {
	s.data = mat
}

func (s *set) Data() *mat.M32f {
	return s.data
}

func (s *set) Normalize(initialMin, initialMax, min, max float32) *set {
	normalize(s.data, initialMin, initialMax, min, max)

	return s
}

func (s *set) OneHot(depth int) *set {
	s.data = oneHotEncode(s.data, depth)

	return s
}

func (s *set) BatchSize() int {
	return s.batchSize
}

func (s *set) BatchShape() mat.Shape {
	return mat.Dim(s.batchSize, s.Data().Shape().D[1])
}

func (s *set) SetBatchSize(batchSize int) *set {
	s.batchSize = batchSize
	return s
}

func (s *set) NumBatches() int {
	return s.data.Shape().D[0] / s.BatchSize()
}

func (s *set) HasNextBatch() bool {
	result := s.batchCounter < s.NumBatches()
	if !result {
		s.batchCounter = 0
	}
	return result
}

func (s *set) ResetBatchCounter() {
	s.batchCounter = 0
}

func (s *set) NextBatch() *mat.M32f {
	if !s.HasNextBatch() {
		s.ResetBatchCounter()
	}

	if s.batchCounter == 0 && s.shouldShuffle {
		s.shuffleData()
	}

	s.batchCounter++

	return s.data.Slice(mat.Dim((s.batchCounter-1)*s.batchSize), mat.Dim(s.batchSize))
}

func (s *set) BatchCounter() int {
	return s.batchCounter
}

func (s *set) shuffleData() {
	matX := s.data.Data()
	matY := s.data.Data()
	rand.Shuffle(s.data.Shape().D[0]*s.data.Shape().D[1], func(i, j int) {
		matX[i], matX[j] = matX[j], matX[i]
		matY[i], matY[j] = matY[i], matY[i]
	})
}

func (s *set) Shuffle() *set {
	s.shouldShuffle = true
	return s
}

func (s *set) DisableShuffle() *set {
	s.shouldShuffle = false
	return s
}

func (s *set) Shape() mat.Shape {
	return s.data.Shape()
}
