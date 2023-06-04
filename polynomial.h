#include "matrix.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <stack>
#include <vector>
#include <type_traits>

#ifndef POLYNOMIAL_LIBRARY_H
#define POLYNOMIAL_LIBRARY_H

namespace polynomial{
    template <typename T>
    class Polynomial{
        public:
            Polynomial(){
                /* default constructor */
                this->degree = 0;
                this->coef = std::vector<T>(1, T());
            }

            Polynomial(std::vector<T>& coefficients){
                /*
                 * constructor
                 * @param degree: the degree of the polynomial
                 * @param coefficients: the coefficients of the polynomial, in increasing order of degree, starting with a constant term
                 * You must include zero coefficients for any terms that are not included in the polynomial, EX: x^2 = {0, 0, 1}
                 * It is also VERY IMPORTANT that the size of the array is equal to the degree + 1
                 */
                this->degree = coefficients.size() - 1;
                this->coef = std::vector<T>(coefficients);
                // if the leading coeficient is 0, reduce the degree
                reduceDegree();
            }

            Polynomial(std::vector<T>&& coefficients){
                /*
                 * constructor
                 * @param degree: the degree of the polynomial
                 * @param coefficients: the coefficients of the polynomial, in increasing order of degree, starting with a constant term
                 * You must include zero coefficients for any terms that are not included in the polynomial, EX: x^2 = {0, 0, 1}
                 * It is also VERY IMPORTANT that the size of the array is equal to the degree + 1
                 */
                this->degree = coefficients.size() - 1;
                this->coef = std::vector<T>(std::move(coefficients));
                // if the leading coeficient is 0, reduce the degree
                reduceDegree();
            }

            Polynomial(const Polynomial& other){
                /*
                 * copy constructor, creates a deep copy of Polynomial other
                 * @param other: the polynomial to copy
                 */
                this->degree = other.degree;
                this->coef = std::vector(other.coef);
                reduceDegree(); // call reduce degree to ensure that the leading coefficient is not 0
            }

            Polynomial(Polynomial&& other){
                /*
                 * copy constructor, creates a deep copy of Polynomial other
                 * @param other: the polynomial to copy
                 */
                this->degree = other.degree;
                this->coef = std::vector(std::move(other.coef));
                reduceDegree(); // call reduce degree to ensure that the leading coefficient is not 0
                other.clear();
            }

            Polynomial(std::vector<T>& x, std::vector<T>& y, size_t degree){
                /*
                 * constructor that fits a polynomial of degree degree to the data points (x[i], y[i]) using Least Squares Regression
                 * @param x: the x coordinates of the points
                 * @param y: the y coordinates of the points
                 * @param degree: the degree of the polynomial
                 */

                if (x.size() != y.size()){
                    throw std::invalid_argument("x and y must be the same size");
                }

                if (x.size() <= degree){
                    throw std::invalid_argument("Not enough data points to fit a polynomial of degree " + std::to_string(degree));
                }

                this->degree = degree;
                this->coef = std::vector<T>(degree + 1, T());

                matrix::Matrix<T> X(x.size(), degree + 1);
                // populate the matrix X
                for (size_t row = 0; row < x.size(); row++){
                    for (size_t col = 0; col <= degree; col++){
                        X(row, col) = pow(x[row], col);
                    }
                }

                matrix::Matrix X_t = X.transpose();
                matrix::Matrix res = X_t * X;

                res = res.inverse();
                res = res * X_t;

                // compute the product of res and Y, don't need a matrix for Y since its just one column
                for (size_t row = 0; row <= degree; row++){
                    T sum = T();
                    for (size_t col = 0; col < x.size(); col++){
                        sum += res(row, col) * y[col];
                    }
                    this->coef[row] = sum;
                }
                reduceDegree(); // always call reduce degree just to be safe, having leading 0s is bad
            }
            ~Polynomial(){
                // dont need to do anything here, the vector will handle memory management
            }

            Polynomial& operator=(const Polynomial& other){
                /*
                 * assignment operator, creates a deep copy of Polynomial other
                 * @param other: the polynomial to copy
                 * @return: a reference to the this object
                 */
                if(this != &other){
                    this->degree = other.degree;
                    this->coef = std::vector(other.coef);
                }
                reduceDegree(); // always call reduce degree just to be safe
                return *this;
            }

            Polynomial& operator=(Polynomial&& other){
                /*
                 * assignment operator, creates a deep copy of Polynomial other
                 * @param other: the polynomial to copy
                 * @return: a reference to the this object
                 */
                if(this != &other){
                    this->degree = other.degree;
                    this->coef = std::vector(std::move(other.coef));
                }
                reduceDegree(); // always call reduce degree just to be safe

                other.clear();

                return *this;
            }

            Polynomial operator+(const Polynomial& other){
                /*
                 * addition operator, adds two polynomials
                 * @param other: the polynomial to add to this polynomial
                 * @return: a new polynomial that is the sum of this polynomial and other, with degree max(this->degree, other.degree)
                 */
                size_t max_degree = this->degree > other.degree ? this->degree : other.degree;
                Polynomial result(max_degree);
                for(size_t i = 0; i <= max_degree; i++){
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
                size_t max_degree = this->degree > other.degree ? this->degree : other.degree;
                Polynomial result(max_degree);
                for(size_t i = 0; i <= max_degree; i++){
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
                size_t max_degree = this->degree + other.degree;
                Polynomial result(max_degree);
                for(size_t i = 0; i <= this->degree; i++){
                    for(size_t j = 0; j <= other.degree; j++){
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
                    if (std::is_arithmetic<T>::value && other.coef[0] == 0){
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
                    T coef = remainder.coef[remainder.degree] / other.coef[other.degree];
                    size_t d = remainder.degree - other.degree;
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
                Polynomial div = *this / other;
                Polynomial result = *this - (div * other);
            }

            Polynomial operator-(){
                /*
                 * unary negation operator, negates all the coefficients of the polynomial
                 * @return: a new polynomial that is the negation of this polynomial
                 */
                Polynomial result(*this);
                for(size_t i = 0; i <= this->degree; i++){
                    result.coef[i] = -result.coef[i];
                }
                return result;
            }

            Polynomial operator+(T scalar){
                /*
                 * addition operator, adds a scalar to a polynomial
                 * @param scalar: the scalar to add to this polynomial
                 * @return: a new polynomial that is the sum of this polynomial and scalar, with degree this->degree
                 */
                Polynomial result(*this);
                result.coef[0] += scalar;
                return result;
            }

            Polynomial operator-(T scalar){
                /*
                 * subtraction operator, subtracts a scalar from a polynomial
                 * @param scalar: the scalar to subtract from this polynomial
                 * @return: a new polynomial that is the difference of this polynomial and scalar, with degree this->degree
                 */
                Polynomial result(*this);
                result.coef[0] -= scalar;
                return result;
            }

            Polynomial operator*(T scalar){
                /*
                 * multiplication operator, multiplies a polynomial by a scalar
                 * @param scalar: the scalar to multiply this polynomial by
                 * @return: a new polynomial that is the product of this polynomial and scalar, with degree this->degree
                 */
                // if scalar is 0 there is a special case where the resulting degree is 0
                if (std::is_arithmetic<T>::value && scalar == 0){
                    return Polynomial();
                }

                Polynomial result(*this);
                for(size_t i = 0; i <= this->degree; i++){
                    result.coef[i] *= scalar;
                }
                return result;
            }

            Polynomial operator/(T scalar){
                /*
                 * division operator, divides a polynomial by a scalar
                 * @throws: std::invalid_argument if scalar is equal to 0, can't divide by 0
                 * @param scalar: the scalar to divide this polynomial by
                 * @return: a new polynomial with all the coefficients of this polynomial divided by scalar
                 */
                if(std::is_arithmetic<T>::value && scalar == 0){
                    throw std::invalid_argument("Cannot divide by 0");
                }
                Polynomial result(*this);
                for(size_t i = 0; i <= this->degree; i++){
                    result.coef[i] /= scalar;
                }
                return result;
            }

            Polynomial& operator+=(T scalar){
                /*
                 * addition assignment operator, adds a scalar to a polynomial
                 * @param scalar: the scalar to add to this polynomial
                 * @return: a reference to this polynomial
                 */
                this->coef[0] += scalar;
                return *this;
            }

            Polynomial& operator-=(T scalar){
                /*
                 * subtraction assignment operator, subtracts a scalar from a polynomial
                 * @param scalar: the scalar to subtract from this polynomial
                 * @return: a reference to this polynomial
                 */
                this->coef[0] -= scalar;
                return *this;
            }

            Polynomial& operator*=(T scalar){
                /*
                 * multiplication assignment operator, multiplies a polynomial by a scalar
                 * @param scalar: the scalar to multiply this polynomial by
                 * @return: a reference to this polynomial
                 */
                // if scalar == 0, there is a special case where the new degree is 0
                if (std::is_arithmetic<T>::value && scalar == 0){
                    this->degree = 0;
                    this->coef = std::vector<T>(1, T());
                    return *this;
                }

                for(size_t i = 0; i <= this->degree; i++){
                    this->coef[i] *= scalar;
                }
                return *this;
            }

            Polynomial& operator/=(T scalar){
                /*
                 * division assignment operator, divides a polynomial by a scalar
                 * @throws: std::invalid_argument if scalar is equal to 0, can't divide by 0
                 * @param scalar: the scalar to divide this polynomial by
                 * @return: a reference to this polynomial
                 */
                if(std::is_arithmetic<T>::value && scalar == 0){
                    throw std::invalid_argument("Cannot divide by 0");
                }

                for(size_t i = 0; i <= this->degree; i++){
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
                size_t max_degree = this->degree > other.degree ? this->degree : other.degree;
                std::vector<T> new_coef(max_degree + 1, T());
                for(size_t i = 0; i <= max_degree; i++){
                    // getCoefficient returns 0 if the degree is too high, so we dont need bounds checking here
                    new_coef[i] = this->getCoefficient(i) + other.getCoefficient(i);
                }

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
                size_t max_degree = this->degree > other.degree ? this->degree : other.degree;
                std::vector<T> new_coef(max_degree + 1, T());
                for(size_t i = 0; i <= max_degree; i++){
                    // getCoefficient returns 0 if the degree is too high, so we dont need bounds checking here
                    new_coef[i] = this->getCoefficient(i) - other.getCoefficient(i);
                }

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
                for(size_t i = 0; i <= this->degree; i++){
                    for(size_t j = 0; j <= other.degree; j++){
                        // binomial expansion theorem, can't just multiply the individual coefficients
                        result.coef[i + j] += this->coef[i] * other.coef[j];
                    }
                }

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
                    this->coef = std::vector<T>(1, T());
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

            std::vector<T> roots(){
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
                std::vector<T> poly_roots;

                if (degree == 1){
                    // special case, polynomial is linear, only one root
                    poly_roots.push_back(-this->coef[0] / this->coef[1]);
                    return poly_roots;
                }

                // handle arbitrary degree case
                // first, find all the derivatives of the function
                std::vector<Polynomial<T>> derivatives(this->degree);

                derivatives[0] = derivative();
                for(size_t i = 1; i < this->degree; i++){
                    derivatives[i] = derivatives[i-1].derivative();
                }

                // now, find the roots of the polynomial using Budan's Theorem
                // we will attempt to isolation regions containing one root, starting with one large region
                std::stack<double> left_bounds;
                std::stack<double> right_bounds;
                // using two seperate stacks instead of one stack containing pairs of doubles to save memory
                left_bounds.push(-1000000.0);
                right_bounds.push(10000000.0);

                // minimum width of an interval, this can be adjusted, lower precision = faster
                const double budan_precision = 1e-2;
                const double newton_precision = 1e-10;
                while(!left_bounds.empty()){
                    double left = left_bounds.top();
                    left_bounds.pop();
                    double right = right_bounds.top();
                    right_bounds.pop();

                    int left_sign_switches = 0;
                    int right_sign_switches = 0;

                    double last_left = operator()(left);
                    double last_right = operator()(right);

                    for (size_t i = 0; i < this->degree; i++){
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
                        if (right - left < budan_precision){
                            // interval is small enough, we can add to the list of root regions
                            poly_roots.push_back((T) newtonRaphson((left + right) / 2.0, newton_precision)); // cast to (T) before adding to vector
                        }else{
                            // interval is too large, split it in half and try again
                            left_bounds.push(left);
                            right_bounds.push((left + right) / 2.0);
                            left_bounds.push((left + right) / 2.0);
                            right_bounds.push(right);
                            continue;
                        }
                    }else if (s == 2){
                        if (right - left < budan_precision){ // complex roots occur in pairs, this is probably a complex root
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
                return poly_roots;
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
                for(size_t i = 1; i <= this->degree; i++){
                    result.coef[i - 1] = this->coef[i] * i;
                }
                return result;
            }

            Polynomial integral(){
                /* Returns the indefinite integral of the polynomial 
                 * @return: a polynomial representing the indefinite integral of this polynomial
                 */
                Polynomial result(this->degree + 1);
                for(size_t i = 0; i <= this->degree; i++){
                    result.coef[i + 1] = this->coef[i] / (i + 1);
                }
                return result;
            }

            Polynomial integral(T lower_bound){
                /* Returns the indefinite integral of the polynomial 
                 * @param lower_bound allows the user to specify a lower bound for the integral,
                 * which shifts the polynomial by that amount, such that the integral evaluated at
                 * the lower bound is equal to 0
                 * @return: a polynomial representing the indefinite integral of this polynomial with lower bound lower_bound
                 */
                Polynomial result(this->degree + 1);
                for(size_t i = 0; i <= this->degree; i++){
                    result.coef[i + 1] = this->coef[i] / (i + 1);
                }
                result.coef[0] = -this->operator()(lower_bound); // shift the integral by the lower bound
                return result;
            }

            T integral(T lower_bound, T upper_bound){
                /* Returns the definite integral of the polynomial 
                 * @param lower_bound: the lower bound of the integral
                 * @param upper_bound: the upper bound of the integral
                 * @return: the definite integral of the polynomial from lower_bound to upper_bound
                 */
                Polynomial result = integral();
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
                const double EPSILON = 0.000001;
                for(size_t i = 0; i <= this->degree; i++){
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

            T operator()(T x){
                /* 
                 * Evaluates the polynomial at a given point
                 * @param x: the point to evaluate the polynomial at
                 * @return: the value of the polynomial at x
                 */
                T result = 0;
                for(size_t i = 0; i <= this->degree; i++){
                    result += this->coef[i] * pow(x, i);
                }
                return result;
            }

            T operator[](size_t index) const{
                /* 
                 * Returns the coefficient of the term of degree index
                 * @param index: the degree of the term to return the coefficient of
                 * @return: the coefficient of the term of degree index, this is not a reference,
                 * if you want to modify the coefficient, use the set_coef method
                 */
                return this->getCoefficient(index);
            }

            size_t getDegree() const{
                // getter for the degree of the polynomial
                return this->degree;
            }

            T getCoefficient(int index) const{
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

            void setCoefficient(size_t index, T value){
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
                    std::vector<T> new_coef(index + 1, T());
                    for(size_t i = 0; i <= this->degree; i++){
                        new_coef[i] = this->coef[i];
                    }
                    for(size_t i = this->degree + 1; i < index; i++){
                        new_coef[i] = 0;
                    }
                    new_coef[index] = value;

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
                for (size_t i = degree-1; i > 0; i--){
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
            size_t degree;
            std::vector<T> coef;

            void clear(){
                degree = 0;
                coef = std::vector<T>(1, T());
            }

            void reduceDegree(){
                /* 
                 * Reduces the degree of the polynomial by removing any trailing 0 coefficients
                 * This method is called after every operation that may reduce the degree of the polynomial
                 */
                while(this->coef[this->degree] == 0 && this->degree > 0){
                    this->coef.pop_back();
                    this->degree--;
                }
            }

            Polynomial(size_t degree){
                /* Initialize a polynomial of degree D, then fill every spot with 0. This is private because behavior is needed 
                 * for other class methods, but should not be used in general because leading zeros coefficients are not allowed
                */
                this->degree = degree;
                this->coef = std::vector<T>(degree + 1, T());
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