# MathLite
MathLite is a lightweight header-only C++ Math Library with classes for Matricies, Linear Algebra, and Polynomials, including Least Squares Regression and Root Finding Algorithims

## Installation Instructions
As MathLite is a header-only Library it requires no installation or external libraries, you can simply download or copy/paste the files into your working directory and #include them. 

## Polynomial Basics
#### Instantiation
There are two ways to create a Polynomial objects. First, you can specify a degree and pass in a double array of coefficients, in order from lowest to highest degree.
```
double coefs[] = {1.0, 2.0, 3.0, 4.0};
polynomial::Polynomial p(3, coefs); // 4x^3 + 3x^2 + 2x + 1
```
Secondly, you can fit a polynomial to a dataset using [Least Squares Regression](https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html). This is done by passing in a degree, the number of datapoints, and two double arrays for the x and y values.
```
double x[] = {1.0, 2.0, 5.0, 10.0, 15.0};
double y[] = {1.0, 8.0, 125.0, 1000.0, 3375.0};
polynomial::Polynomial p(x, y, 5, 3); // x^3
```
#### Evaluation
To evaluate a Polynomial at a specific point, you can just use ()
```
double coefs[] = {1.0, 2.0, 3.0, 4.0};
polynomial::Polynomial p(3, coefs); // 4x^3 + 3x^2 + 2x + 1
p(2.5); // 87.25
```
#### Arithmetic
The Polynomial class has every type of arithmetic you could think of, including polynomial long division and polynomial modulus. See the source code for a full list of arithmetic available. 
#### Root Finding
The Polynomial class has a root finding algorithim built in. It uses a combination of the [Budan-Fourier Theorem](https://www.tandfonline.com/doi/pdf/10.1080/00029890.1943.11991462?needAccess=true) and the [Newton-Raphson Method](https://web.mit.edu/10.001/Web/Course_Notes/NLAE/node6.html) to approximate Polynomial roots to an arbitrary precision. The value of the precision can be easily changed by modifying the source code, but the default precision is 1e-10, which should be sufficient for any use case. Polynomial.roots() returns a double array of size degree, however some or all of these values may be NaN if the polynomial has less than degree real roots. The real roots will always come first in the array, followed by NaNs.
```
double arr[] = {-0.1, 0, 1.89, 0.0, -2.9, 0.0, 10.1, -9.22};
polynomial::Polynomial p1(7, arr);
double* roots = p1.roots();
for (int i = 0; i < 7; i++){
    std::cout << roots[i] << " ";
}
// 0.978233 0.239041 -0.238016 nan nan nan nan
```
#### Calculus
The Polynomial class has methods for finding the derivative of a polynomial, finding the indefinte integral starting at a lower bound, and definite integration from lower to upper bounds.
```
double arr[] = {3.0, 0.0, 1.0, 0.0, 2.0}; // 2x^4 + x^2 + 3
polynomial::Polynomial p1(4, arr);
std::cout << p1.integral(-5.0, 5.0) << std::endl; // 2613.33
```
