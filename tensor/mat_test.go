package tensor

import (
	"testing"
)

func Test_mat_equals_whenTrueMultiDimension(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0}, {2.0}}
	m2 := [][]float64{{1.0, 2.0}, {2.0}}

	result := equals(m1, m2)

	if !result {
		t.Fatal("matrices should be equal")
	}
}

func Test_mat_equals_whenFalseBecauseOfDifferentXLength(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0}, {2.0}}
	m2 := [][]float64{{1.0, 2.0}}

	result := equals(m1, m2)

	if result {
		t.Fatal("matrices should not be equal")
	}
}

func Test_mat_equals_whenFalseBecauseOfDifferentYLength(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0}, {2.0}}
	m2 := [][]float64{{1.0, 2.0}, {2.0, 3.0}}

	result := equals(m1, m2)

	if result {
		t.Fatal("matrices should not be equal")
	}
}

func Test_mat_add_singleDimension(t *testing.T) {
	m1 := [][]float64{{1.0, 2.0, 3.0}}
	m2 := [][]float64{{1.0, 2.0, 3.0}}
	
	result := add(m1, m2)
	
	if len(result) != 1 {
		t.Fatalf("expected %d, got %d", 1, len(result))
	}
	if result[0][0] != 2.0 || result[0][1] != 4.0 || result[0][2] != 6.0 {
        t.Fatal("matrices don't match")
	}
}

func Test_mat_add_multiDimension(t *testing.T) {
	m1 := [][]float64{{1.0}, {2.0}}
	m2 := [][]float64{{1.0}, {2.0}}

	result := add(m1, m2)

	if len(result) != 2 {
		t.Fatalf("expected %d, got %d", 2, len(result))
	}
	if result[0][0] != 2.0 || result[1][0] != 4.0 {
		t.Fatal("matrices don't match")
	}
}

func Test_mat_remove_multiDimension(t *testing.T) {
	m1 := [][]float64{{1.0}, {2.0}}
	m2 := [][]float64{{1.0}, {2.0}}

	result := sub(m1, m2)

	if len(result) != 2 {
		t.Fatalf("expected %d, got %d", 2, len(result))
	}
	if result[0][0] != 0.0 || result[1][0] != 0.0 {
		t.Fatal("matrices don't match")
	}
}

func Test_mat_normalize(t *testing.T) {
	mat := [][]float64{{2.0, 3.0}}

    result := normalize(mat)

	if result[0][0] != 0.0 || result[0][1] != 1.0 {
		t.Fatal("matrix is not normalized")
	}
}

func Test_mat_dot_multiDimension(t *testing.T) {
	m1 := [][]float64{{1, 0}, {0, 1}}
	m2 := [][]float64{{4, 1}, {2, 2}}

	result := dot(m1, m2)

	expected := [][]float64{{4, 1}, {2, 2}}
	if !equals(result, expected) {
		t.Fatal("dot matrix is incorrect")
	}
}

func Test_mat_dot_multiDimensionOfDifferentSizes(t *testing.T) {
	m1 := [][]float64{{1, 0, 3}, {0, 1, 2}}
	m2 := [][]float64{{4, 1}, {2, 2}, {1, 5}}

	result := dot(m1, m2)

	expected := [][]float64{{7, 16}, {4, 12}}
	if !equals(result, expected) {
		t.Fatal("dot matrix is incorrect")
	}
}

func Test_mat_sum0_multiDimension(t *testing.T) {
	mat := [][]float64{{1, 0, 3}, {0, 1, 2}}

	result := sum(mat, 0)

	expected := [][]float64{{1, 1, 5}}
	if !equals(result, expected) {
		t.Fatal("matrices are not equals")
	}
}

func Test_mat_sum1_multiDimension(t *testing.T) {
	mat := [][]float64{{1, 0, 3}, {0, 1, 2}}

	result := sum(mat, 1)

	expected := [][]float64{{4}, {3}}
	if !equals(result, expected) {
		t.Fatal("matrices are not equals")
	}
}

func Test_mat_expand0_multiDimension(t *testing.T) {
	mat := [][]float64{{1, 0, 3}}

	result := expand(mat, 0, 3)

	expected := [][]float64{{1, 0, 3}, {1, 0, 3}, {1, 0, 3}}
	if !equals(result, expected) {
		t.Fatal("matrices are not equals")
	}
}

func Test_mat_expand1_multiDimension(t *testing.T) {
	mat := [][]float64{{1}, {2}}

	result := expand(mat, 1, 3)

	expected := [][]float64{{1, 1, 1}, {2, 2, 2}}
	if !equals(result, expected) {
		t.Fatal("matrices are not equals")
	}
}
