// IMP is the small imperative language from the K framework tutorial.
// Arithmetic expressions are integers, program variables, addition,
// division, and parenthesized expressions. Boolean expressions are the
// literals true and false, a less-than-or-equal comparison of arithmetic
// expressions, boolean negation, and conjunction. Statements are: a block
// (a possibly-empty statement enclosed in curly braces), variable
// assignment terminated by a semicolon, sequential composition of two
// statements, an if-then-else conditional, and a while loop. A program is
// a sequence of integer variable declarations (int x, y, z;) followed by a
// statement. This is the current K (7.x) syntax.

module IMP-SYNTAX
  imports INT-SYNTAX
  imports BOOL-SYNTAX
  imports ID-SYNTAX
