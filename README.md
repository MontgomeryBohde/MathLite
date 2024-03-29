# MathLite
MathLite is a lightweight header-only C++ Math Library with classes for Matricies, Linear Algebra, and Polynomials, including Least Squares Regression, Root Finding Algorithims, and Calculus

## Installation Instructions
As MathLite is a header-only Library it requires no installation or external libraries, you can simply download or copy/paste the files into your working directory and #include them. 

## Polynomial Basics
#### Instantiation
There are two ways to create a Polynomial objects. First, you can specify a degree and pass in a double array of coefficients, in order from lowest to highest degree.
```
vector<double> coefs = {1.0, 2.0, 3.0, 4.0};
polynomial::Polynomial<double> p(coefs); // 4x^3 + 3x^2 + 2x + 1
```
Secondly, you can fit a polynomial to a dataset using [Least Squares Regression](https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html). This is done by passing in a degree and two vectors for the x/ y values.
```
vector<double> x = {1.0, 2.0, 5.0, 10.0, 15.0};
vector<double> y = {1.0, 8.0, 125.0, 1000.0, 3375.0};
polynomial::Polynomial<double> p(x, y, 3); // x^3
```
#### Evaluation
To evaluate a Polynomial at a specific point, you can just use ()
```
vector<double> coefs = {1.0, 2.0, 3.0, 4.0};
polynomial::Polynomial<double> p(coefs); // 4x^3 + 3x^2 + 2x + 1
p(2.5); // 87.25
```
#### Arithmetic
The Polynomial class has every type of arithmetic you could think of, including polynomial long division and polynomial modulus. See the source code for a full list of arithmetic available. 
#### Root Finding
The Polynomial class has a root finding algorithim built in. It uses a combination of the [Budan-Fourier Theorem](https://www.tandfonline.com/doi/pdf/10.1080/00029890.1943.11991462?needAccess=true) and the [Newton-Raphson Method](https://web.mit.edu/10.001/Web/Course_Notes/NLAE/node6.html) to approximate Polynomial roots to arbitrary precision. The value of the precision can be easily changed by modifying the source code, but the default precision is 1e-10, which should be sufficient for any use case. 
```
vector<double> coefs = {-0.1, 0, 1.89, 0.0, -2.9, 0.0, 10.1, -9.22};
polynomial::Polynomial<double> p1(coefs);
vector<double> roots = p1.roots();
for (auto root: roots){
    std::cout << root << " ";
}
// 0.978233 0.239041 -0.238016
```
#### Calculus
The Polynomial class has methods for finding the derivative of a polynomial, finding the indefinte integral starting at a lower bound, and definite integration from lower to upper bounds.
```
vector<double> coefs = {3.0, 0.0, 1.0, 0.0, 2.0}; // 2x^4 + x^2 + 3
polynomial::Polynomial<double> p1(4, coefs);
std::cout << p1.integral(-5.0, 5.0); // 2613.33
```
## Matrix Basics
#### Instantiation
You can create a matrix in one of three ways, you can initialize an empty N x M matrix, an N x M matrix filled with values from a vector, or an N x M matrix filled with a default value.
```
vector<int> values = {1, 2, 3, 4, 5, 6};
matrix::Matrix<int> m1(2, 3, values);
matrix::Matrix<int> m2(2, 3);
matrix::Matrix<int> m3(2, 3, 0);
```
#### Functionality
The Matrix Library is a quite extensive Linear Algebra Library and has functions for scalar arithmetic, Matrix addition and subtraction, Matrix Multiplication, Matrix Augmentation, Matrix Exponentation, Matrix Transposition, calculating the determinant of a Matrix, finding the Reduced Row Echelon Form of a Matrix, finding the rank of a Matrix, finding the inverse of a Matrix, and more. See the source code for full documentation. Note that some of these algorithms are not optimized, and could be done faster with a more complicated implementation. 

#### Printing
You can print a nicely formated version of a Matrix to standard output by calling the .print() method
````
vector<int> values = {1, 2, 3, 4, 5, 6};
matrix::Matrix mat(2, 3, values);
mat.print();
// [1 2 3]
// [4 5 6]
````
