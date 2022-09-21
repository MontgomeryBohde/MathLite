#include "matrix.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <stack>

#ifndef POLYNOMIAL_LIBRARY_H
#define POLYNOMIAL_LIBRARY_H

namespace polynomial{
    class Polynomial{
        public:
            Polynomial(){
                /* default constructor */
                this->degree = 0;
                this->coef = new double[1];
                this->coef[0] = 0;
            }
            Polynomial(int degree, double* coefficients){
                /*
                 * constructor
                 * @param degree: the degree of the polynomial
                 * @param coefficients: the coefficients of the polynomial, in increasing order of degree, starting with a constant term
                 * You must include zero coefficients for any terms that are not included in the polynomial, EX: x^2 = {0, 0, 1}
                 * It is also VERY IMPORTANT that the size of the array is equal to the degree + 1
                 */
                this->degree = degree;
                this->coef = new double[degree + 1];
                for(int i = 0; i <= degree; i++){
                    this->coef[i] = coefficients[i];
                }
                // if the leading coeficient is 0, reduce the degree
                reduceDegree();
            }
            Polynomial(const Polynomial& other){
                /*
                 * copy constructor, creates a deep copy of Polynomial other
                 * @param other: the polynomial to copy
                 */
                this->degree = other.degree;
                this->coef = new double[other.degree + 1];
                for(int i = 0; i <= other.degree; i++){
                    this->coef[i] = other.coef[i];
                }
                reduceDegree(); // call reduce degree to ensure that the leading coefficient is not 0
            }
            Polynomial(double* x, double* y, int arr_size, int degree){
                /*
                 * constructor that fits a polynomial of degree degree to the data points (x[i], y[i]) using Least Squares Regression
                 * @param x: the x coordinates of the points
                 * @param y: the y coordinates of the points
                 * @param arr_size: the size of the arrays x and y, IMPORTANT that this is equal to the amount of points in the arrays, and is more than degree
                 * @param degree: the degree of the polynomial
                 */
                this->degree = degree;
                this->coef = new double[degree + 1];

                matrix::Matrix X(arr_size, degree + 1);
                // populate the matrix X
                for (int row = 0; row < arr_size; row++){
                    X(row, 0) = 1;
                    for (int col = 1; col <= degree; col++){
                        X(row, col) = pow(x[row], col);
                    }
                }
                matrix::Matrix X_t = X.transpose();
                matrix::Matrix res = X_t * X;
                res = res.inverse();
                res = res * X_t;
                res.print();
                std::cout << std::endl;
                // compute the product of res and Y, don't need a matrix for Y since its just one column
                for (int row = 0; row <= degree; row++){
                    double sum = 0;
                    for (int col = 0; col < arr_size; col++){
                        sum += res(row, col) * y[col];
                    }
                    this->coef[row] = sum;
                }
                reduceDegree(); // always call reduce degree just to be safe, having leading 0s is bad
            }
            ~Polynomial(){
                /* destructor */
                delete[] this->coef;
            }
            Polynomial& operator=(const Polynomial& other){
                /*
                 * assignment operator, creates a deep copy of Polynomial other
                 * @param other: the polynomial to copy
                 * @return: a reference to the this object
                 */
                if(this != &other){
                    delete[] this->coef;
                    this->degree = other.degree;
                    this->coef = new double[other.degree + 1];
                    for(int i = 0; i <= other.degree; i++){
                        this->coef[i] = other.coef[i];
                    }
                }
                reduceDegree(); // always call reduce degree just to be safe
                return *this;
            }
            Polynomial operator+(const Polynomial& other){
                /*
                 * addition operator, adds two polynomials
                 * @param other: the polynomial to add to this polynomial
                 * @return: a new polynomial that is the sum of this polynomial and other, with degree max(this->degree, other.degree)
                 */
                int max_degree = this->degree > other.degree ? this->degree : other.degree;
                Polynomial result(max_degree);
                for(int i = 0; i <= max_degree; i++){
                    // getCoefficient returns 0 if the degree is too high, so we dont need bounds checking here
                    result.coef[i] = this->getCoefficient(i) + other.getCoefficient(i);
                }
                // if the leading coeficient is 0, reduce the degree
                result.reduceDegree();
                return result;
            }
            Polynomial operator-(const Polynomial& other){
                /*
                 * subtraction operator, subtracts two polynomials
                 * @param other: the polynomial to subtract from this polynomial
                 * @return: a new polynomial that is the difference of this polynomial and other, with degree max(this->degree, other.degree)
                 */
                int max_degree = this->degree > other.degree ? this->degree : other.degree;
                Polynomial result(max_degree);
                for(int i = 0; i <= max_degree; i++){
                    if(i <= this->degree && i <= other.degree){
                        result.coef[i] = this->coef[i] - other.coef[i];
                    }
                    else if(i <= this->degree){
                        result.coef[i] = this->coef[i];
                    }
                    else{
                        result.coef[i] = -other.coef[i];
                    }
                }
                // if the leading coeficient is 0, reduce the degree
                result.reduceDegree();
                return result;
            }
            Polynomial operator*(const Polynomial& other){
                /*
                 * multiplication operator, multiplies two polynomials
                 * @param other: the polynomial to multiply this polynomial by
                 * @return: a new polynomial that is the product of this polynomial and other, with degree this->degree + other.degree
                 */
                int max_degree = this->degree + other.degree;
                Polynomial result(max_degree);
                for(int i = 0; i <= this->degree; i++){
                    for(int j = 0; j <= other.degree; j++){
                        // binomial expansion theorem, can't just multiply the individual coefficients
                        result.coef[i + j] += this->coef[i] * other.coef[j];
                    }
                }
                return result;
            }
            Polynomial operator/(const Polynomial& other){
                /*
                 * division operator, divides two polynomials with no remainder
                 * @throws: std::invalid_argument if other is a constant and is equal to 0, can't divide by 0
                 * @param other: the polynomial to divide this polynomial by
                 * @return: a new polynomial that is the quotient of this polynomial and other, with degree this->degree - other.degree
                 */
                // if the polynomial is degree 0, we are just dividing by a constant
                if(other.degree == 0){
                    if (other.coef[0] == 0){
                        throw std::invalid_argument("Cannot divide by 0");
                    }
                    return *this / other.coef[0];
                }
                // return a 0 degree polynomial if the divisor is larger than the dividend
                if (this->degree < other.degree){
                    return Polynomial();
                }
                Polynomial result(this->degree - other.degree);
                Polynomial remainder(*this);
                while(remainder.degree >= other.degree){
                    double coef = remainder.coef[remainder.degree] / other.coef[other.degree];
                    int d = remainder.degree - other.degree;
                    Polynomial temp(d);
                    temp.coef[d] = coef;
                    result += temp;
                    remainder = remainder - (temp * other);
                }
                return result;
            }
            Polynomial operator%(const Polynomial& other){
                /*
                 * modulo operator, divides two polynomials and returns the remainder
                 * if the polynomial is degree 0, we just mod each coefficient by the constant
                 * If Degree(other) > Degree(this), then the result is just a copy of this
                 * @throws: std::invalid_argument if other is a constant and is equal to 0, can't divide by 0
                 * @param other: the polynomial to divide this polynomial by
                 * @return: a new polynomial that is the remainder of this polynomial and other
                 */
                // if the polynomial is degree 0, we are just dividing by a constant
                if(other.degree == 0){
                    if (other.coef[0] == 0){
                        throw std::invalid_argument("Cannot divide by 0");
                    }
                    // if taking the mod of a polynomial by a constant, just the polynomial with all the coefficients modded
                    Polynomial result(*this);
                    for(int i = 0; i <= this->degree; i++){
                        result.coef[i] = fmod(result.coef[i], other.coef[0]);
                    }
                    return result;
                }
                // return a 0 degree polynomial if the divisor is larger than the dividend
                if (this->degree < other.degree){
                    return Polynomial(*this);
                }
                Polynomial remainder(*this);
                while(remainder.degree >= other.degree){
                    double coef = remainder.coef[remainder.degree] / other.coef[other.degree];
                    int d = remainder.degree - other.degree;
                    Polynomial temp(d);
                    temp.coef[d] = coef;
                    remainder = remainder - (temp * other);
                }
                return remainder;
            }
            Polynomial operator-(){
                /*
                 * unary negation operator, negates all the coefficients of the polynomial
                 * @return: a new polynomial that is the negation of this polynomial
                 */
                Polynomial result(*this);
                for(int i = 0; i <= this->degree; i++){
                    result.coef[i] = -result.coef[i];
                }
                return result;
            }
            Polynomial operator+(double scalar){
                /*
                 * addition operator, adds a scalar to a polynomial
                 * @param scalar: the scalar to add to this polynomial
                 * @return: a new polynomial that is the sum of this polynomial and scalar, with degree this->degree
                 */
                Polynomial result(*this);
                result.coef[0] += scalar;
                return result;
            }
            Polynomial operator-(double scalar){
                /*
                 * subtraction operator, subtracts a scalar from a polynomial
                 * @param scalar: the scalar to subtract from this polynomial
                 * @return: a new polynomial that is the difference of this polynomial and scalar, with degree this->degree
                 */
                Polynomial result(*this);
                result.coef[0] -= scalar;
                return result;
            }
            Polynomial operator*(double scalar){
                /*
                 * multiplication operator, multiplies a polynomial by a scalar
                 * @param scalar: the scalar to multiply this polynomial by
                 * @return: a new polynomial that is the product of this polynomial and scalar, with degree this->degree
                 */
                // if scalar is 0 there is a special case where the resulting degree is 0
                if (scalar == 0){
                    Polynomial result(0);
                    result.coef[0] = 0;
                    return result;
                }
                Polynomial result(*this);
                for(int i = 0; i <= this->degree; i++){
                    result.coef[i] *= scalar;
                }
                return result;
            }
            Polynomial operator/(double scalar){
                /*
                 * division operator, divides a polynomial by a scalar
                 * @throws: std::invalid_argument if scalar is equal to 0, can't divide by 0
                 * @param scalar: the scalar to divide this polynomial by
                 * @return: a new polynomial with all the coefficients of this polynomial divided by scalar
                 */
                if(scalar == 0){
                    throw std::invalid_argument("Cannot divide by 0");
                }
                Polynomial result(*this);
                for(int i = 0; i <= this->degree; i++){
                    result.coef[i] /= scalar;
                }
                return result;
            }
            Polynomial& operator+=(double scalar){
                /*
                 * addition assignment operator, adds a scalar to a polynomial
                 * @param scalar: the scalar to add to this polynomial
                 * @return: a reference to this polynomial
                 */
                this->coef[0] += scalar;
                return *this;
            }
            Polynomial& operator-=(double scalar){
                /*
                 * subtraction assignment operator, subtracts a scalar from a polynomial
                 * @param scalar: the scalar to subtract from this polynomial
                 * @return: a reference to this polynomial
                 */
                this->coef[0] -= scalar;
                return *this;
            }
            Polynomial& operator*=(double scalar){
                /*
                 * multiplication assignment operator, multiplies a polynomial by a scalar
                 * @param scalar: the scalar to multiply this polynomial by
                 * @return: a reference to this polynomial
                 */
                // if scalar == 0, there is a special case where the new degree is 0
                if (scalar == 0){
                    this->degree = 0;
                    delete[] this->coef;
                    this->coef = new double[1];
                    this->coef[0] = 0;
                    return *this;
                }
                for(int i = 0; i <= this->degree; i++){
                    this->coef[i] *= scalar;
                }
                return *this;
            }
            Polynomial& operator/=(double scalar){
                /*
                 * division assignment operator, divides a polynomial by a scalar
                 * @throws: std::invalid_argument if scalar is equal to 0, can't divide by 0
                 * @param scalar: the scalar to divide this polynomial by
                 * @return: a reference to this polynomial
                 */
                if(scalar == 0){
                    throw std::invalid_argument("Cannot divide by 0");
                }
                for(int i = 0; i <= this->degree; i++){
                    this->coef[i] /= scalar;
                }
                return *this;
            }
            Polynomial& operator+=(const Polynomial& other){
                /*
                 * addition assignment operator, adds a polynomial to this polynomial
                 * This will result in increasing the degree of this polynomial if other.degree > this->degree
                 * It's also possible that the degree with be reduced if the leading coefficient of the result is 0
                 * @param other: the polynomial to add to this polynomial
                 * @return: a reference to this polynomial
                 */
                int max_degree = this->degree > other.degree ? this->degree : other.degree;
                double* new_coef = new double[max_degree + 1];
                for(int i = 0; i <= max_degree; i++){
                    // getCoefficient returns 0 if the degree is too high, so we dont need bounds checking here
                    new_coef[i] = this->getCoefficient(i) + other.getCoefficient(i);
                }

                // if the leading coeficient is 0, reduce the degree
                delete[] this->coef;
                this->coef = new_coef;
                this->degree = max_degree;
                this->reduceDegree();

                return *this;
            }
            Polynomial& operator-=(const Polynomial& other){
                /*
                 * subtraction assignment operator, subtracts a polynomial from this polynomial
                 * This will result in increasing the degree of this polynomial if other.degree > this->degree
                 * It's also possible that the degree with be reduced if the leading coefficient of the result is 0
                 * @param other: the polynomial to subtract from this polynomial
                 * @return: a reference to this polynomial
                 */
                int max_degree = this->degree > other.degree ? this->degree : other.degree;
                double* new_coef = new double[max_degree + 1];
                for(int i = 0; i <= max_degree; i++){
                    // getCoefficient returns 0 if the degree is too high, so we dont need bounds checking here
                    new_coef[i] = this->getCoefficient(i) - other.getCoefficient(i);
                }

                // if the leading coeficient is 0, reduce the degree
                delete[] this->coef;
                this->coef = new_coef;
                this->degree = max_degree;
                this->reduceDegree();

                return *this;
            }
            Polynomial& operator*=(const Polynomial& other){
                /*
                 * multiplication assignment operator, multiplies a polynomial by another polynomial
                 * @param other: the polynomial to multiply this polynomial by
                 * @return: a reference to this polynomial
                 */
                Polynomial result(this->degree + other.degree);
                for(int i = 0; i <= this->degree; i++){
                    for(int j = 0; j <= other.degree; j++){
                        // binomial expansion theorem, can't just multiply the individual coefficients
                        result.coef[i + j] += this->coef[i] * other.coef[j];
                    }
                }
                delete[] this->coef;
                this->coef = result.coef;
                this->degree = result.degree;
                return *this;
            }
            Polynomial& operator/=(const Polynomial& other){
                /*
                 * division assignment operator, divides a polynomial by another polynomial
                 * @param other: the polynomial to divide this polynomial by
                 * @return: a reference to this polynomial
                 */
                if (this->degree < other.degree){
                    // special case when the degree of the divisor is greater than the dividend, result is 0
                    this->degree = 0;
                    delete[] this->coef;
                    this->coef = new double[1];
                    this->coef[0] = 0;
                    return *this;
                }
                *this = *this / other;
                return *this;
            }
            Polynomial& operator%=(const Polynomial& other){
                /*
                 * modulo assignment operator, finds the remainder of a polynomial divided by another polynomial
                 * @param other: the polynomial to divide this polynomial by
                 * @return: a reference to this polynomial
                 */
                Polynomial result = *this / other;
                *this = *this - (result * other);
                return *this;
            }
            double* roots(){
                /* Computes the roots of a polynomial using a combination of Budan's Theorem 
                 * and the Newton-Raphson method
                 * @throws: std::invalid_argument if the polynomial is of degree 0 
                 * @return: an array of the roots of the polynomial, with size degree
                 * Important to note that there will be at most degree roots, but there may be less, which will be occupied
                 * by NaN values in the result array 
                */
               if (this->degree == 0){
                // special case, polynomial is a constant, throw an error because we can't declare
                // an array of size 0
                    throw std::invalid_argument("Cannot find roots of a constant");
               }
               // a polynomial can have at most degree roots
                double* roots = new double[this->degree];
                for (int i = 0; i < this->degree; i++){
                    roots[i] = std::nan("Not a real root");
                }
                int roots_idx = 0; // keep track of where we are in the roots array
                if (degree == 1){
                    // special case, polynomial is linear, only one root
                    roots[0] = -this->coef[0] / this->coef[1];
                    return roots;
                }
                // handle arbitrary degree case
                // first, find all the derivatives of the function
                Polynomial* derivatives = new Polynomial[this->degree];
                derivatives[0] = derivative();
                for(int i = 1; i < this->degree; i++){
                    derivatives[i] = derivatives[i-1].derivative();
                }
                // now, find the roots of the polynomial using Budan's Theorem
                // we will attempt to isolation regions containing one root, starting with one large region
                std::stack<double> left_bounds;
                std::stack<double> right_bounds;
                // using two seperate stacks instead of one stack containing pairs of doubles to save memory
                left_bounds.push((double) std::numeric_limits<long int>::min()+1000);
                right_bounds.push((double) std::numeric_limits<long int>::max());

                // minimum width of an interval, this can be adjusted, lower precision = faster
                const double precision = 1e-2;

                while(!left_bounds.empty()){
                    double left = left_bounds.top();
                    left_bounds.pop();
                    double right = right_bounds.top();
                    right_bounds.pop();
                    int left_sign_switches = 0;
                    int right_sign_switches = 0;
                    double last_left = operator()(left);
                    double last_right = operator()(right);
                    for (int i = 0; i < this->degree; i++){
                        // iterate through the derivatives, counting the number of sign switches
                        // in the interval
                        double left_val = derivatives[i](left);
                        double right_val = derivatives[i](right);
                        if (left_val * last_left < 0){
                            left_sign_switches++; 
                            last_left = left_val;
                        }
                        if (right_val * last_right < 0){
                            right_sign_switches++;
                            last_right = right_val;
                        }
                    }
                    int s = left_sign_switches - right_sign_switches;
                    if (s == 0){
                        // no sign switches in the interval, no roots
                        continue;
                    }else if (s == 1){ // one root in the interval means it must be real, but stil want to decrease interval
                        if (right - left < precision){
                            // interval is small enough, we can add to the list of root regions
                            roots[roots_idx] = newtonRaphson((left + right) / 2.0, 1e-10);
                            roots_idx++;
                        }else{
                            // interval is too large, split it in half and try again
                            left_bounds.push(left);
                            right_bounds.push((left + right) / 2);
                            left_bounds.push((left + right) / 2);
                            right_bounds.push(right);
                            continue;
                        }
                    }else if (s == 2){
                        if (right - left < precision){ // complex roots occur in pairs, this is probably a complex root
                            continue;
                        }else{ // split interval in two and add to stack
                            left_bounds.push(left);
                            right_bounds.push((left + right) / 2.0);
                            left_bounds.push((left + right) / 2.0);
                            right_bounds.push(right);
                        }
                    }else{ // split interval in two and add to stack
                        left_bounds.push(left);
                        right_bounds.push((left + right) / 2.0);
                        left_bounds.push((left + right) / 2.0);
                        right_bounds.push(right);
                    }
                }
                return roots;
            }
            Polynomial derivative(){
                /*
                 * Computes the derivative of a polynomial
                 * @return: a polynomial representing the derivative of this polynomial
                 */
                if(this->degree == 0){
                    // special case, derivative of a constant is 0
                    Polynomial result(0);
                    result.coef[0] = 0;
                    return result;
                }
                Polynomial result(this->degree - 1);
                for(int i = 1; i <= this->degree; i++){
                    result.coef[i - 1] = this->coef[i] * i;
                }
                return result;
            }
            Polynomial indef_integral(double lower_bound = 0.0){
                /* Returns the indefinite integral of the polynomial 
                 * @param lower_bound allows the user to specify a lower bound for the integral,
                 * which shifts the polynomial by that amount, such that the integral evaluated at
                 * the lower bound is equal to 0
                 * @return: a polynomial representing the indefinite integral of this polynomial with lower bound lower_bound
                 */
                Polynomial result(this->degree + 1);
                for(int i = 0; i <= this->degree; i++){
                    result.coef[i + 1] = this->coef[i] / (i + 1);
                }
                result.coef[0] = -this->operator()(lower_bound); // shift the integral by the lower bound
                return result;
            }
            double def_integral(double lower_bound, double upper_bound){
                /* Returns the definite integral of the polynomial 
                 * @param lower_bound: the lower bound of the integral
                 * @param upper_bound: the upper bound of the integral
                 * @return: the definite integral of the polynomial from lower_bound to upper_bound
                 */
                Polynomial result(this->degree + 1);
                for(int i = 0; i <= this->degree; i++){
                    result.coef[i + 1] = this->coef[i] / (i + 1);
                }
                return result(upper_bound) - result(lower_bound);
            }
            bool operator==(const Polynomial& other){
                /* 
                 * Checks if two polynomials are equal
                 * @param other: the polynomial to compare to
                 * @return: true if all the coefficients are equal, false otherwise
                 */
                if(this->degree != other.degree){
                    return false;
                }
                // we need to check if coefs are within a tolerance because of floating point errors
                // this value of epsilon is chosen arbitrarily and you may want to consider altering it based on your use case
                // e.g. if working with polynomials with very large coefficients, you may want to increase epsilon
                const double EPSILON = 0.0000001;
                for(int i = 0; i <= this->degree; i++){
                    if(fabs(coef[i] - other.coef[i]) > EPSILON){
                        return false;
                    }
                }
                return true;
            }
            bool operator!=(const Polynomial& other){
                // self explanatory
                return !(*this == other);
            }
            double operator()(double x){
                /* 
                 * Evaluates the polynomial at a given point
                 * @param x: the point to evaluate the polynomial at
                 * @return: the value of the polynomial at x
                 */
                double result = 0;
                for(int i = 0; i <= this->degree; i++){
                    result += this->coef[i] * pow(x, i);
                }
                return result;
            }
            double operator[](int index) const{
                /* 
                 * Returns the coefficient of the term of degree index
                 * @param index: the degree of the term to return the coefficient of
                 * @return: the coefficient of the term of degree index, this is not a reference,
                 * if you want to modify the coefficient, use the set_coef method
                 */
                if(index < 0){
                    throw std::out_of_range("Degree must be non-negative");
                }
                else if(index > this->degree){
                    // don't want to raise an error if user tries to get the coefficient of a term with degree > than the this->degree because
                    // the degree of the polynomial may be reduced by operations like +=, -=, *=, etc., causing unforseen errors
                    return 0;
                }
                return this->coef[index];
            }
            int getDegree() const{
                // getter for the degree of the polynomial
                return this->degree;
            }
            double getCoefficient(int index) const{
                /*
                 * Returns the coefficient of the term of degree index
                 * @param index: the degree of the term to return the coefficient of
                 * @return: the coefficient of the term of degree index
                 */
                if(index < 0){
                    throw std::out_of_range("Degree must be non-negative");
                }
                else if(index > this->degree){
                    // don't want to raise an error if user tries to get the coefficient of a term with degree > than the this->degree because
                    // the degree of the polynomial may be reduced by operations like +=, -=, *=, etc., causing unforseen errors
                    return 0;
                }
                return this->coef[index];
            }
            void setCoefficient(int index, double value){
                /* Set the coefficient of the term with the specified degree 
                 * EX: If you wanted to set the coefficient of x^3, you would call setCoefficient(3, value)
                 * If you wanted to set the coefficient of the constant term, you would call setCoefficient(0, value)
                 * This method will also update the degree of the polynomial if the specified degree is
                 * greater than the current degree
                 * @param index: the degree of the term to set the coefficient of
                 * @param value: the value to set the coefficient to
                 */
                if (index > this->degree){
                    // if index greater than degree, we will need to create a new coef array
                    double* new_coef = new double[index + 1];
                    for(int i = 0; i <= this->degree; i++){
                        new_coef[i] = this->coef[i];
                    }
                    for(int i = this->degree + 1; i < index; i++){
                        new_coef[i] = 0;
                    }
                    new_coef[index] = value;
                    // make sure to free up the memory from the old array before reassigning
                    delete[] this->coef;
                    this->coef = new_coef;
                    this->degree = index;
                }
                else{
                    this->coef[degree] = value;
                }
                return;
            }
            void print() const{
                /* Print the polynomial to std output 
                 * Any terms with a 0 coefficient will not be printed
                */
                if (degree == 0){
                    std::cout << coef[0] << std::endl;
                    return;
                }
                std::cout << coef[degree] << "x^" << degree;
                for (int i = degree-1; i > 0; i--){
                    if (coef[i] == 0){
                        continue;
                    }
                    if (coef[i] > 0){
                        std::cout << " + ";
                    }else{
                        std::cout << " - ";
                    }
                    std::cout << fabs(coef[i]) << "x^" << i;
                }
                if (coef[0] > 0){
                    std::cout << " + " << coef[0] << std::endl;
                }else if (coef[0] < 0){
                    std::cout << " - " << fabs(coef[0]) << std::endl;
                }else{
                    std::cout << std::endl;
                }
                return;
            }
        private:
            int degree;
            double* coef;
            void reduceDegree(){
                /* 
                 * Reduces the degree of the polynomial by removing any trailing 0 coefficients
                 * This method is called after every operation that may reduce the degree of the polynomial
                 */
                while(this->coef[this->degree] == 0 && this->degree > 0){
                    this->degree--;
                }
            }
            Polynomial(int degree){
                /* Initialize a polynomial of degree D, then fill every spot with 0. This is private because behavior is needed 
                 * for other class methods, but should not be used in general because leading zeros coefficients are not allowed
                */
                this->degree = degree;
                this->coef = new double[degree + 1];
                for (int i = 0; i <= degree; i++){
                    this->coef[i] = 0;
                }
            }
            double newtonRaphson(double x0, double epsilon){
                /* 
                 * Uses Newton-Raphson method to find the root of the polynomial
                 * @param x0: the initial guess for the root
                 * @param epsilon: the error tolerance
                 * @return: the root of the polynomial
                 */
                Polynomial derivative = this->derivative();
                double x1 = x0 - (this->operator()(x0) / derivative(x0));
                while(fabs(x1 - x0) > epsilon){
                    x0 = x1;
                    x1 = x0 - (this->operator()(x0) / derivative(x0));
                }
                return x1;
            }
    };
}
#endif