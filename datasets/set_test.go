package datasets

import (
	"github.com/stretchr/testify/assert"
	"ml-framework/mat"
	"testing"
)

func TestSet_BatchSize_hasTheCorrectValue(t *testing.T) {
	s := newSet(5)
	s.SetData(mat.NewMat32fZeros(mat.Dim(15)))

	assert.Equal(t, 5, s.BatchSize())
}

func TestSet_NumBatches(t *testing.T) {
	s := newSet(5)
	s.SetData(mat.NewMat32fZeros(mat.Dim(15)))

	assert.Equal(t, 3, s.NumBatches())
}

func TestSet_NextBatch_firstBatch(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(9), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}))

	result := s.NextBatch()

	assert.Equal(t, mat.Dim(3).D, result.Shape().D)
	assert.Equal(t, []float32{1, 2, 3}, result.Data())
}

func TestSet_NextBatch_secondBatch(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(9), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}))

	_ = s.NextBatch()
	result := s.NextBatch()

	assert.Equal(t, mat.Dim(3).D, result.Shape().D)
	assert.Equal(t, []float32{4, 5, 6}, result.Data())
}

func TestSet_NextBatch_lastBatch(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(9), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}))

	_ = s.NextBatch()
	_ = s.NextBatch()
	result := s.NextBatch()

	assert.Equal(t, mat.Dim(3).D, result.Shape().D)
	assert.Equal(t, []float32{7, 8, 9}, result.Data())
}

func TestSet_NextBatch_lastBatchWhenDataDoesNotAlign(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(7), []float32{
		1, 2, 3,
		4, 5, 6,
		7,
	}))

	_ = s.NextBatch()
	_ = s.NextBatch()
	result := s.NextBatch()

	assert.Equal(t, []float32{1, 2, 3}, result.Data())
}

func TestSet_NextBatch_tryingToAccessBatchAfterLast(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(9), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}))

	_ = s.NextBatch()
	_ = s.NextBatch()
	_ = s.NextBatch()
	result := s.NextBatch()

	assert.Equal(t, []float32{1, 2, 3}, result.Data())
}

func TestSet_HasNextBatch_whenTrue(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(6), []float32{
		1, 2, 3,
		4, 5, 6,
	}))
	_ = s.NextBatch()

	result := s.HasNextBatch()

	assert.True(t, result)
}

func TestSet_HasNextBatch_whenFalse(t *testing.T) {
	s := newSet(3)
	s.SetData(mat.FromSlice32f(mat.Dim(6), []float32{
		1, 2, 3,
		4, 5, 6,
	}))
	_ = s.NextBatch()
	_ = s.NextBatch()

	result := s.HasNextBatch()

	assert.False(t, result)
}
