package tensor

type Shape struct {
	X int
	Y int
}

func (s Shape) ToArray() []int {
	return []int{s.X, s.Y}
}

func (s Shape) Size() int {
	return s.X * s.Y
}

func (s Shape) Equals(other Shape) bool {
	return s.X == other.X && s.Y == other.Y
}