package mat

func Normalize32f(data []float32) {
	var min float32
	var max float32
	for i := 0; i < len(data); i++ {
		if data[i] < min {
			min = data[i]
		}

		if data[i] > max {
			max = data[i]
		}
	}

	for i := 0; i < len(data); i++ {
		data[i] = (data[i] - min) / (max - min)
	}
}
