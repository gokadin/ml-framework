package mat

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_IndexOf_oneDimZeroIndex(t *testing.T) {
	shape := Dim(3)

	index := shape.IndexOf(0)

	assert.Equal(t, 0, index)
}

func Test_IndexOf_oneDimNonZeroIndex(t *testing.T) {
	shape := Dim(3)

	index := shape.IndexOf(2)

	assert.Equal(t, 2, index)
}

func Test_IndexOf_twoDimsZeroIndices(t *testing.T) {
	shape := Dim(3, 4)

	index := shape.IndexOf(0, 0)

	assert.Equal(t, 0, index)
}

func Test_IndexOf_multiDimsZeroIndicesMissingAnArgument(t *testing.T) {
	shape := Dim(2, 3, 4)

	index := shape.IndexOf(0)

	assert.Equal(t, 0, index)
}

func Test_IndexOf_multiDimsNonZeroIndices(t *testing.T) {
	shape := Dim(2, 3, 4)

	index := shape.IndexOf(1, 1, 1)

	assert.Equal(t, 17, index)
}

func Test_IndexOf_multiDimsNonZeroIndicesVaried(t *testing.T) {
	shape := Dim(2, 3, 4)

	index := shape.IndexOf(1, 2, 2)

	assert.Equal(t, 22, index)
}
