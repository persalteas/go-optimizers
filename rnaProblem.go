package main

import "math"

// AtomType values are types of atoms that you can find in a nucleotide (P, C1', N1...)
type AtomType uint8

// The various types of atoms that you can find in a nucleotide
const (
	P   AtomType = iota // Phosphate
	OP1                 // Phosphate
	OP2                 // Phosphate
	C1P                 // Ribose
	C2P                 // Ribose
	C3P                 // Ribose
	C4P                 // Ribose
	C5P                 // Ribose
	O2P                 // Ribose
	O3P                 // Ribose
	O4P                 // Ribose
	O5P                 // Ribose
	C2                  // G base, C base, U base, A base
	C4                  // G base, C base, U base, A base
	C5                  // G base, C base, U base, A base
	C6                  // G base, C base, U base, A base
	C8                  // G base, A base
	O2                  // C base, U base
	O4                  // U base
	O6                  // G base
	N1                  // G base, C base, U base, A base
	N2                  // G base
	N3                  // G base, C base, U base, A base
	N4                  // C base
	N6                  // A base
	N7                  // G base, A base
	N9                  // G base, A base
)

// NucleotideType values A, C, G, or U
type NucleotideType uint8

// NucleotideType values A, C, G, or U
const (
	A NucleotideType = iota
	C
	G
	U
	T
)

// Structure is the base data structure containing information about an RNA conformation
type Structure struct {
	nBases   uint
	nAtoms   uint
	sequence []NucleotideType
	coords   []float32
}

func (s *Structure) getAtomCoords(ntPosition uint, a AtomType) []float32 {
	var index = 78*(ntPosition-1) + uint(a)
	return s.coords[index : index+2]
}

func (s *Structure) setAtomCoords(ntPosition uint, a AtomType, x, y, z float32) {
	var index = 78*(ntPosition-1) + uint(a)
	s.coords[index] = x
	s.coords[index+1] = y
	s.coords[index+2] = z
}

func (s *Structure) getNucleotideCenter(pos uint) []float32 {
	var xyz []float32
	switch nt := s.sequence[pos]; nt {
	case A, G: // purines, return N1
		xyz = s.getAtomCoords(pos, N1)
	case C, U, T: // pyrimidines, return N3
		xyz = s.getAtomCoords(pos, N3)
	}
	return xyz
}

func getAtomDistanceBetween(atom1, atom2 []float32) float64 {
	var dx = float64(atom1[0] - atom2[0])
	var dy = float64(atom1[1] - atom2[1])
	var dz = float64(atom1[2] - atom2[2])
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func (s *Structure) getResidueDistanceBetween(pos1, pos2 uint) float64 {
	return getAtomDistanceBetween(s.getNucleotideCenter(pos1), s.getNucleotideCenter(pos2))
}

func getCosAngleBetween(atom1, atom2, atom3 []float32) float64 {
	// Par le théorème d'Al Kashi, une fois développé.
	// atom2 should be the angle's tip
	var x1 = atom1[0]
	var y1 = atom1[1]
	var z1 = atom1[2]
	var x2 = atom2[0]
	var y2 = atom2[1]
	var z2 = atom2[2]
	var x3 = atom3[0]
	var y3 = atom3[1]
	var z3 = atom3[2]
	var x11 = x1 * x1
	var x12 = x1 * x2
	var x13 = x1 * x3
	var y11 = y1 * y1
	var y12 = y1 * y2
	var y13 = y1 * y3
	var z11 = z1 * z1
	var z12 = z1 * z2
	var z13 = z1 * z3
	var x22 = x2 * x2
	var x23 = x2 * x3
	var y22 = y2 * y2
	var y23 = y2 * y3
	var z22 = z2 * z2
	var z23 = z2 * z3
	var x33 = x3 * x3
	var y33 = y3 * y3
	var z33 = z3 * z3
	var numerator = x22 + y22 + z22 + x13 + y13 + z13 - x12 - y12 - z12 - x23 - y23 - z23
	var denominator = (x22 - 2*x12 + x11 + y22 - 2*y12 + y11 + z22 - 2*z12 + z11) * (x33 - 2*x13 + x11 + y33 - 2*y13 + y11 + z33 - 2*z13 + z11)
	return float64(numerator) / math.Sqrt(float64(denominator))
}

func getCosTorsionBetween(atom1, atom2, atom3, atom4 []float32) float64 {
	// Par le théorème d'Al Kashi, une fois développé.
	var x1 = atom1[0]
	var y1 = atom1[1]
	var z1 = atom1[2]
	var x2 = atom2[0]
	var y2 = atom2[1]
	var z2 = atom2[2]
	var x3 = atom3[0]
	var y3 = atom3[1]
	var z3 = atom3[2]
	var x4 = atom4[0]
	var y4 = atom4[1]
	var z4 = atom4[2]

	// Normal vector to the plane (atom1,atom2,atom3)
	var p1 = []float32{(y2-y1)*(z3-z2) - (z2-z1)*(y3-y2), (z2-z1)*(x3-x2) - (x2-x1)*(z3-z2), (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)}
	// Normal vector to the plane (atom2,atom3,atom4)
	var p2 = []float32{(y3-y2)*(z4-z3) - (z3-z2)*(y4-y3), (z3-z2)*(x4-x3) - (x3-x2)*(z4-z3), (x3-x2)*(y4-y3) - (y3-y2)*(x4-x3)}
	var numerator = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
	var denominator = (p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2]) * (p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2])
	return float64(numerator) / math.Sqrt(float64(denominator))
}

var rnaEq = []string{"", ""}
