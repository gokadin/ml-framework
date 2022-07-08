package mat

type ShapeN struct {
	D []int
}

func Dim(dimensions ...int) ShapeN {
	for _, d := range dimensions {
		if d <= 0 {
			panic("cannot have a negative or size zero dimension")
		}
	}
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

func (s ShapeN) Equals(other ShapeN) bool {
	if len(other.D) != len(s.D) {
		return false
	}

	for i := 0; i < len(s.D); i++ {
		if s.D[i] != other.D[i] {
			return false
		}
	}

	return true
}

// IndexOf
//	for 2 dimensions: index = d[0].index * d[1].size + d[1].index
//	for 3 dimensions: index = (d[0].index * d[1].size * d[2].size) + (d[1].index * d[2].size) + d[2].index
///**
func (s ShapeN) IndexOf(values ...int) (index int) {
	if len(values) == 0 || len(values) > s.Count() {
		panic("invalid argument for IndexOf operation on shape")
	}

	if len(values) < s.Count() {
		for i := 0; i < s.Count()-len(values)+1; i++ {
			values = append(values, 0)
		}
	}

	mul := 1
	for i := s.Count() - 1; i >= 0; i-- {
		if i == s.Count()-1 {
			index += values[i]
		} else {
			mul *= s.D[i+1]
			index += values[i] * mul
		}
	}

	return index
}

func (s ShapeN) Copy() ShapeN {
	dimensions := make([]int, s.Count())
	for i := 0; i < len(dimensions); i++ {
		dimensions[i] = s.D[i]
	}
	return Dim(dimensions...)
}

func AddShapes(a, b ShapeN) ShapeN {
	if a.Count() != b.Count() {
		panic("cannot add shapes of different dimensions")
	}

	result := a.Copy()
	for i := 0; i < a.Count(); i++ {
		result.D[i] = a.D[i] + b.D[i]
	}

	return result
}
