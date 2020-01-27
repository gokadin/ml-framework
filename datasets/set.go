package datasets

type set struct {
	data [][]float64
}

func (s *set) Data() [][]float64 {
	return s.data
}

func (s *set) ShapeX() int {
	return len(s.data)
}

func (s *set) ShapeY() int {
	return len(s.data[0])
}

func (s *set) Normalize(min, max float64) *set {
	normalize(s.data, min, max)

	return s
}

func (s *set) OneHot(depth int) *set {
	oneHotEncode(s.data, depth)

	return s
}
