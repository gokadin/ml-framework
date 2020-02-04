package mat

func CreateMatrix(n, m int) [][]float64 {
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, m)
	}
	return mat
}

func Normalize(a [][]float64) [][]float64 {
	minValue := Min(a)
	maxValue := Max(a)
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

func Max(a [][]float64) float64 {
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

func Min(a [][]float64) float64 {
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
