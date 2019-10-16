package runners

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_mean(t *testing.T) {
	input := []float64{0.0, 0.5, 1.0}

	result := mean(input)

	assert.Equal(t, 0.5, result)
}

func Test_partition_sizeWithFittingNumberOfBatchesOnTheFirstIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 0, 2)

	assert.Equal(t, 2, len(results))
}

func Test_partition_sizeWithFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 2)

	assert.Equal(t, 2, len(results))
}

func Test_partition_sizeWithNonFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 1)

	assert.Equal(t, 1, len(results))
}

func Test_partition_dataWithFittingNumberOfBatchesOnTheFirstIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 0, 2)

	expected := [][]float64{{1}, {2}}
	assert.Equal(t, expected, results)
}

func Test_partition_dataWithFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 2)

	expected := [][]float64{{3}, {4}}
	assert.Equal(t, expected, results)
}

func Test_partition_dataWithNonFittingNumberOfBatchesOnTheSecondIteration(t *testing.T) {
	data := [][]float64{{1.0}, {2.0}, {3.0}, {4.0}}

	results := partitionData(data, 1, 2)

	expected := [][]float64{{3}, {4}}
	assert.Equal(t, expected, results)
}
