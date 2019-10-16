package mat

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_mat_equals_whenTrueMultiDimension(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0}, {2.0}}
	m2 := [][]float64{{1.0, 2.0}, {2.0}}

	assert.Equal(t, m1, m2)
}

func Test_mat_equals_whenFalseBecauseOfDifferentXLength(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0}, {2.0}}
	m2 := [][]float64{{1.0, 2.0}}

	assert.NotEqual(t, m1, m2)
}

func Test_mat_equals_whenFalseBecauseOfDifferentYLength(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0}, {2.0}}
	m2 := [][]float64{{1.0, 2.0}, {2.0, 3.0}}

	assert.NotEqual(t, m1, m2)
}

func Test_mat_add_singleDimension(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0, 3.0}}
	m2 := [][]float64{{1.0, 2.0, 3.0}}
	
	result := Add(m1, m2)

	expected := [][]float64{{2, 4, 6}}
	assert.Equal(t, expected, result)
}

func Test_mat_add_multiDimension(t *testing.T) {
	m1 := [][]float64{{1.0}, {2.0}}
	m2 := [][]float64{{1.0}, {2.0}}

	result := Add(m1, m2)

	expected := [][]float64{{2}, {4}}
	assert.Equal(t, expected, result)
}

func Test_mat_remove_multiDimension(t *testing.T) {
	m1 := [][]float64{{1.0}, {2.0}}
	m2 := [][]float64{{1.0}, {2.0}}

	result := Sub(m1, m2)

	expected := [][]float64{{0}, {0}}
	assert.Equal(t, expected, result)
}

func Test_mat_normalize(t *testing.T) {
	mat := [][]float64{{2.0, 3.0}}

    result := Normalize(mat)

    expected := [][]float64{{0, 1}}
	assert.Equal(t, expected, result)
}

func Test_mat_dot_multiDimension(t *testing.T) {
	m1 := [][]float64{{1, 0}, {0, 1}}
	m2 := [][]float64{{4, 1}, {2, 2}}

	result := Dot(m1, m2)

	expected := [][]float64{{4, 1}, {2, 2}}
	assert.Equal(t, expected, result)
}

func Test_mat_dot_multiDimensionOfDifferentSizes(t *testing.T) {
	m1 := [][]float64{{1, 0, 3}, {0, 1, 2}}
	m2 := [][]float64{{4, 1}, {2, 2}, {1, 5}}

	result := Dot(m1, m2)

	expected := [][]float64{{7, 16}, {4, 12}}
	assert.Equal(t, expected, result)
}

func Test_mat_sum0_multiDimension(t *testing.T) {
	mat := [][]float64{{1, 0, 3}, {0, 1, 2}}

	result := Sum(mat, 0)

	expected := [][]float64{{1, 1, 5}}
	assert.Equal(t, expected, result)
}

func Test_mat_sum1_multiDimension(t *testing.T) {
	mat := [][]float64{{1, 0, 3}, {0, 1, 2}}

	result := Sum(mat, 1)

	expected := [][]float64{{4}, {3}}
	assert.Equal(t, expected, result)
}

func Test_mat_expand0_multiDimension(t *testing.T) {
	mat := [][]float64{{1, 0, 3}}

	result := Expand(mat, 0, 3)

	expected := [][]float64{{1, 0, 3}, {1, 0, 3}, {1, 0, 3}}
	assert.Equal(t, expected, result)
}

func Test_mat_expand1_multiDimension(t *testing.T) {
	mat := [][]float64{{1}, {2}}

	result := Expand(mat, 1, 3)

	expected := [][]float64{{1, 1, 1}, {2, 2, 2}}
	assert.Equal(t, expected, result)
}
