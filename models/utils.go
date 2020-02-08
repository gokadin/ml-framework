package models

func mean(arr []float64) float64 {
	mean := 0.0
    for _, value := range arr {
        mean += value
	}
    return mean / float64(len(arr))
}
