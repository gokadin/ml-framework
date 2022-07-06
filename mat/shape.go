package mat

type ShapeN struct {
	D []int
}

func Dim(dimensions ...int) ShapeN {
	return ShapeN{dimensions}
}

func (s ShapeN) Size() int {
	result := 1
	for _, d := range s.D {
		result *= d
	}
	return result
}

func (s ShapeN) Count() int {
	return len(s.D)
}
