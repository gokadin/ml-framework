package tensor

import (
	"log"
	"math"
)

func add(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func addScalar(a [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + scalar
		}
	}
	return result
}

func divideScalarBy(a [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = scalar / a[i][j]
		}
	}
	return result
}

func sub(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

func subFromScalar(a [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = scalar - a[i][j]
		}
	}
	return result
}

func mul(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] * b[i][j]
		}
	}
	return result
}

func div(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] / b[i][j]
		}
	}
	return result
}

func divScalar(a [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] / scalar
		}
	}
	return result
}

func pow(a [][]float64, power float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = math.Pow(a[i][j], power)
		}
	}
	return result
}

func exp(a [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = math.Pow(math.E, a[i][j])
		}
	}
	return result
}

func dot(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(a[0]) == 0 || len(b) == 0 || len(b[0]) == 0 || len(a[0]) != len(b) {
		log.Fatalf("cannot multiply matrices of incompatible sizes -> %dx%d and %dx%d", len(a), len(a[0]), len(b), len(b[0]))
	}

	resultSizeI := len(a)
	resultSizeJ := len(b[0])

	result := make([][]float64, resultSizeI)
	for i := range a {
		result[i] = make([]float64, resultSizeJ)
		for j := 0; j < len(b[0]); j++ {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

func normalize(a [][]float64) [][]float64 {
	minValue := min(a)
	maxValue := max(a)
	difference := maxValue - minValue
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = (a[i][j] - minValue) / difference
		}
	}
	return result
}

func max(a [][]float64) float64 {
	maxValue := a[0][0]
	for i := range a {
		for j := range a[i] {
			if a[i][j] > maxValue {
				maxValue = a[i][j]
			}
		}
	}
	return maxValue
}

func min(a [][]float64) float64 {
	minValue := a[0][0]
	for i := range a {
		for j := range a[i] {
			if a[i][j] < minValue {
				minValue = a[i][j]
			}
		}
	}
	return minValue
}

func equals(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
    return true
}

func mulScalar(mat [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(mat))
	for i := range mat {
		result[i] = make([]float64, len(mat[i]))
		for j := range mat[i] {
			result[i][j] = mat[i][j] * scalar
		}
	}
	return result
}

func transpose(mat [][]float64) [][]float64 {
	result := make([][]float64, len(mat[0]))
	for i := 0; i < len(result); i++ {
		result[i] = make([]float64, len(mat))
	}

	for i := range mat {
		for j := range mat[i] {
			result[j][i] = mat[i][j]
		}
	}
	return result
}

func sum(mat [][]float64, axis int) [][]float64 {
    switch axis {
	case 0:
		return sum0(mat)
	case 1:
		return sum1(mat)
	default:
		log.Fatal("sum only supports axis 0 and 1")
		return nil
	}
}

func sum0(mat [][]float64) [][]float64 {
	result := make([][]float64, 1)
	result[0] = make([]float64, len(mat[0]))
	for i := range mat {
		for j := range mat[i] {
			result[0][j] += mat[i][j]
		}
	}
	return result
}

func sum1(mat [][]float64) [][]float64 {
	result := make([][]float64, len(mat))
	for i := range mat {
		result[i] = make([]float64, 1)
		for j := range mat[i] {
			result[i][0] += mat[i][j]
		}
	}
	return result
}

func expand(mat [][]float64, axis, copies int) [][]float64 {
	switch axis {
	case 0:
		return expand0(mat, copies)
	case 1:
		return expand1(mat, copies)
	default:
		log.Fatal("expand only supports axis 0 and 1")
		return nil
	}
}

func expand0(mat [][]float64, copies int) [][]float64 {
	if len(mat) != 1 {
		log.Fatalf("incompatible matrix size for expand operation on X axis -> %d", len(mat))
	}

	result := make([][]float64, copies)
	for i := range result {
		result[i] = make([]float64, len(mat[0]))
		for j := range mat[0] {
			result[i][j] = mat[0][j]
		}
	}
	return result
}

func expand1(mat [][]float64, copies int) [][]float64 {
	if len(mat) == 0 || len(mat[0]) != 1 {
		log.Fatalf("incompatible matrix size for expand operation on X axis -> %d", len(mat))
	}

	result := make([][]float64, len(mat))
	for i := range mat {
		result[i] = make([]float64, len(mat[i]) * copies)
		copyCounter := 0
		for j := 0; j < copies; j++ {
			for k := range mat[i] {
				result[i][copyCounter] = mat[i][k]
				copyCounter++
			}
		}
	}
	return result
}
