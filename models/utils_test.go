package models

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_mean(t *testing.T) {
	input := []float64{0.0, 0.5, 1.0}

	result := mean(input)

	assert.Equal(t, 0.5, result)
}
