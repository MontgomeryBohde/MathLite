#include <iostream>
#include <cmath>

#ifndef MATRIX_H
#define MATRIX_H

namespace matrix{
    class Matrix{
        public:
            Matrix(int rows, int cols){
                /*
                 * Constructor for Matrix class, initializes a matrix of size rows x cols without instatiating elements
                 * @param rows: number of rows in the matrix
                 * @param cols: number of columns in the matrix
                 */

                this->rows = rows;
                this->cols = cols;
                // use a 1D array to represent 2D matrix beacuse you can't use non-const value for cols
                this->data = new double[rows * cols];
            }
            Matrix(int rows, int cols, double* data){
                /*
                * Constructor for Matrix class, initializes a matrix of size rows x cols and copys elements from data into the Matrix
                * @param rows: number of rows in the matrix
                * @param cols: number of columns in the matrix
                * @param data: pointer to array of doubles to be copied into the matrix, this MUST be of size rows * cols
                */
                this->rows = rows;
                this->cols = cols;
                // even if data is passed in, we need to create a copy of it to store in memory to prevent deallocation errors later
                this->data = new double[rows * cols];
                for (int i = 0; i < rows*cols; i++){
                    this->data[i] = data[i];
                }
            }
            Matrix(int rows, int cols, double default_value){
                /*
                * Constructor for Matrix class, initializes a matrix of size rows x cols and sets all elements to default_value
                * @param rows: number of rows in the matrix
                * @param cols: number of columns in the matrix
                * @param default_value: value to set all elements in the matrix to
                */
                this->rows = rows;
                this->cols = cols;
                this->data = new double[rows * cols];
                for (int i = 0; i < rows*cols; i++){
                    data[i] = default_value;
                }
            }
            Matrix(const Matrix& other){
                /*
                * Copy constructor for Matrix class, creates a deep copy of the other matrix
                * @param other: Matrix to be copied
                */
                this->rows = other.rows;
                this->cols = other.cols;
                this->data = new double[rows*cols];
                for (int i = 0; i < rows*cols; i++){
                    data[i] = other.data[i];
                }
            }
            ~Matrix(){
                //Destructor for Matrix class, frees memory allocated for the matrix
                delete[] data;
            }
            Matrix& operator=(const Matrix& other){
                /*
                * Assignment operator for Matrix class, creates a deep copy of the other matrix
                * @param other: Matrix to be copied
                */
                this->rows = other.rows;
                this->cols = other.cols;
                // ensure memory is freed before allocating new memory for data
                delete[] this->data;
                this->data = new double[rows*cols];
                for (int i = 0; i < rows*cols; i++){
                    data[i] = other.data[i];
                }
                return *this;
            }
            Matrix operator+(const Matrix& other){
                /*
                * Addition operator for Matrix class, adds two matrices together
                * @throws: std::invalid_argument if the matrices are not the same size
                * @param other: Matrix to be added to this matrix
                * @return: new Matrix that is the sum of this matrix and other
                */
                Matrix new_mat(rows, cols, data);
                if(!(rows == other.rows && cols == other.cols)){
                    throw std::invalid_argument("Matrices Must Have Same Dimensions");
                }
                for (int i = 0; i < rows*cols; i++){
                    new_mat.data[i] = data[i] + other.data[i];
                }
                return new_mat;
            }
            Matrix operator-(const Matrix& other){
                /*
                * Subtraction operator for Matrix class, subtracts two matrices elementwise
                * @throws std::invalid_argument if matrices are not the same size
                * @param other: Matrix to be subtracted from this matrix
                * @return: new Matrix that is the difference of this matrix and other
                */
                Matrix new_mat(rows, cols, data);
                if(!(rows == other.rows && cols == other.cols)){
                    throw std::invalid_argument("Matrices Must Have Same Dimensions");
                }
                for (int i = 0; i < rows*cols; i++){
                    new_mat.data[i] = data[i] - other.data[i];
                }
                return new_mat;
            }
            Matrix operator*(const Matrix& other){
                /*
                * Multiplication operator for Matrix class, multiplies two matrices together
                * @throws std::invalid_argument if the number of columns in this matrix does not equal the number of rows in other
                * @param other: Matrix to be multiplied by this matrix
                * @return: new Matrix that is the product of this matrix and other
                */
                Matrix new_mat(rows, other.cols);
                if(cols != other.rows){
                    throw std::invalid_argument("Incompatible Matrix Dimensions");
                }
                for (int i = 0; i < rows; i++){
                    for (int j = 0; j < other.cols; j++){
                        for (int k = 0; k < cols; k++){
                            new_mat.data[i*other.cols + j] += data[i*cols + k] * other.data[k*other.cols + j];
                        }
                    }
                }
                return new_mat;
            }
            Matrix operator*(double scalar){
                /*
                * Multiplication operator for Matrix class, multiplies a matrix by a scalar
                * @param scalar: scalar to be multiplied by this matrix
                * @return: new Matrix that is the product of this matrix and scalar
                */
                Matrix new_mat(rows, cols, data);
                for (int i = 0; i < rows*cols; i++){
                    new_mat.data[i] *= scalar;
                }
                return new_mat;
            }
            Matrix operator/(double scalar){
                /*
                * Division operator for Matrix class, divides a matrix by a scalar
                * @throws invalid_argument if scalar is 0
                * @param scalar: scalar to be divided by this matrix
                * @return: new Matrix that is the quotient of this matrix and scalar
                */
                if (scalar == 0){
                    throw std::invalid_argument("Cannot Divide By Zero");
                }
                Matrix new_mat(rows, cols, data);
                for (int i = 0; i < rows*cols; i++){
                    new_mat.data[i] /= scalar;
                }
                return new_mat;
            }
            Matrix operator+(double scalar){
                /*
                * Addition operator for Matrix class, adds a scalar to every element in the matrix
                * @param scalar: scalar to be added to this matrix
                * @return: new Matrix that is the sum of this matrix and scalar
                */
                Matrix new_mat(rows, cols, data);
                for (int i = 0; i < rows*cols; i++){
                    new_mat.data[i] += scalar;
                }
                return new_mat;
            }
            Matrix operator-(double scalar){
                /*
                * Subtraction operator for Matrix class, subtracts a scalar from every element in the matrix
                * @param scalar: scalar to be subtracted from this matrix
                * @return: new Matrix that is the difference of this matrix and scalar
                */
                Matrix new_mat(rows, cols, data);
                for (int i = 0; i < rows*cols; i++){
                    new_mat.data[i] -= scalar;
                }
                return new_mat;
            }   
            Matrix& operator+=(const Matrix& other){
                /*
                * Addition assignment operator for Matrix class, adds two matrices together, elementwise
                * @throws invalid_argument if matrices do not have the same dimensions
                * @param other: Matrix to be added to this matrix
                * @return: reference to this matrix
                */
                if(!(rows == other.rows && cols == other.cols)){
                    throw std::invalid_argument("Matrices Must Have Same Dimensions");
                }
                 for(int i = 0; i < rows * cols; i++){
                    data[i] += other.data[i];
                }
                return *this;
            }
            Matrix& operator-=(const Matrix& other){
                /*
                * Subtraction assignment operator for Matrix class, subtracts two matrices together, elementwise
                * @throws invalid_argument if matrices do not have the same dimensions
                * @param other: Matrix to be subtracted from this matrix
                * @return: reference to this matrix
                */
                if(!(rows == other.rows && cols == other.cols)){
                    throw std::invalid_argument("Matrices Must Have Same Dimensions");
                }
                 for(int i = 0; i < rows * cols; i++){
                    data[i] -= other.data[i];
                }
                return *this;
            }
            Matrix& operator*=(const Matrix& other){
                /*
                * Multiplication assignment operator for Matrix class, multiplies two matrices together
                * @throws invalid_argument if the number of columns in this matrix does not equal the number of rows in other
                * @param other: Matrix to be multiplied by this matrix
                * @return: reference to this matrix
                */
                if(cols != other.rows){
                    throw std::invalid_argument("Incompatible Matrix Dimensions");
                }
                Matrix new_mat(rows, other.cols);
                for (int i = 0; i < rows; i++){
                    for (int j = 0; j < other.cols; j++){
                        for (int k = 0; k < cols; k++){
                            new_mat.data[i*other.cols + j] += data[i*cols + k] * other.data[k*other.cols + j];
                        }
                    }
                }
                *this = new_mat;
                return *this;
            }
            Matrix& operator*=(double scalar){
                /*
                * Multiplication assignment operator for Matrix class, multiplies a matrix by a scalar, elementwise
                * @param scalar: scalar to be multiplied by this matrix
                * @return: reference to this matrix
                */
                for(int i = 0; i < rows * cols; i++){
                    data[i] *= scalar;
                }
                return *this;
            }
            Matrix& operator/=(double scalar){
                /*
                * Division assignment operator for Matrix class, divides a matrix by a scalar, elementwise
                * @throws invalid_argument if scalar is 0
                * @param scalar: scalar to be divided by this matrix
                * @return: reference to this matrix
                */
                if (scalar == 0){
                    throw std::invalid_argument("Cannot Divide By Zero");
                }
                for(int i = 0; i < rows * cols; i++){
                    data[i] /= scalar;
                }
                return *this;
            }
            Matrix operator^(int power){
                /*
                * Power operator for Matrix class, raises a matrix to a power
                * If the power == -1, the inverse of the matrix is returned
                * @throws invalid_argument if the matrix is not square
                * @param power: power to raise this matrix to
                * @return: new Matrix that is this matrix raised to the power
                */
                if(rows != cols){
                    throw std::invalid_argument("Matrix Must Be Square");
                }
                if (power == 0){
                    Matrix new_mat(rows, cols);
                    for (int i = 0; i < rows; i++){
                        new_mat.data[i*cols + i] = 1;
                    }
                    return new_mat;
                }
                if (power < 0){ // if the power is negative, first find the inverse matrix and exponentiate that
                    return inverse() ^ (-power);
                }
                if (power == 1){
                    return Matrix(rows, cols, data);
                }else{
                    Matrix new_mat = operator^(power/2);
                    if (power % 2 == 0){
                        return new_mat * new_mat;
                    }else{
                        return new_mat * new_mat * *this;
                    }
                }    
            }
            Matrix transpose(){
                /*
                * Transpose operator for Matrix class, returns the transpose of a matrix
                * @return: new Matrix that is the transpose of this matrix
                */
                Matrix new_mat(cols, rows);
                for(int new_row = 0; new_row < cols; new_row++){
                    for (int new_col = 0; new_col < rows; new_col++){
                        new_mat.data[new_row * rows + new_col] = data[new_col * cols + new_row];
                    }
                }
                return new_mat;
            }
            double det(){
                /*
                * Determinant operator for Matrix class, returns the determinant of a matrix
                * @throws invalid_argument if the matrix is not square
                * @return: double that is the determinant of this matrix
                */
                if(rows != cols){
                    throw std::invalid_argument("Matrix Must Be Square");
                }
                if (rows == 1){ // handle simple cases
                    return data[0];
                }else if (rows == 2){
                    return data[0] * data[3] - data[1] * data[2];
                }else{ // handle arbitrary NxN cases
                    double det = 0;
                    for (int i = 0; i < rows; i++){
                        // create submatrix then find its determinant and either add or subtract it from the total
                        Matrix new_mat(rows - 1, cols - 1);
                        for (int new_row = 0; new_row < rows - 1; new_row++){
                            for (int new_col = 0; new_col < cols - 1; new_col++){
                                if (new_col < i){
                                    new_mat.data[new_row * (cols - 1) + new_col] = data[(new_row + 1) * cols + new_col];
                                }else{
                                    new_mat.data[new_row * (cols - 1) + new_col] = data[(new_row + 1) * cols + new_col + 1];
                                }
                            }
                        }
                        if (i & 1){ // if i is odd
                            det -= data[i] * new_mat.det();
                        }else{
                            det += data[i] * new_mat.det();
                        }
                    }
                    return det;
                }
            }
            int rank(){
                /*
                * Rank operator for Matrix class, returns the rank of a matrix
                * @return: int representing the rank of the matrix
                */
                Matrix new_mat = rref();
                int rank = 0;
                for (int i = 0; i < rows; i++){
                    for (int j = 0; j < cols; j++){
                        // only increment rank if the row is not all zeros, e.g. linearly independent
                        if (new_mat.data[i * cols + j] != 0){
                            rank++;
                            break;
                        }
                    }
                }
                return rank;
            }
            Matrix rref(){
                /*
                * Reduced Row Echelon Form operator for Matrix class, returns the reduced row echelon form of a matrix
                * @throws invalid_argument if the matrix is has more rows than columns
                * @return: new Matrix that is the reduced row echelon form of this matrix
                */
                Matrix new_mat(rows, cols, data);
                int lead = 0;
                for (int r = 0; r < rows; r++){
                    if (lead >= cols){
                        throw std::invalid_argument("Matrix Must Not Have More Rows");
                    }
                    int i = r;
                    while (new_mat.data[i * cols + lead] == 0){
                        i++;
                        if (i == rows){
                            i = r;
                            lead++;
                            if (lead == cols){
                                return new_mat;
                            }
                        }
                    }
                    for (int j = 0; j < cols; j++){
                        double temp = new_mat.data[r * cols + j];
                        new_mat.data[r * cols + j] = new_mat.data[i * cols + j];
                        new_mat.data[i * cols + j] = temp;
                    }
                    double div = new_mat.data[r * cols + lead];
                    for (int j = 0; j < cols; j++){
                        new_mat.data[r * cols + j] /= div;
                    }
                    for (int j = 0; j < rows; j++){
                        if (j != r){
                            double mult = new_mat.data[j * cols + lead];
                            for (int k = 0; k < cols; k++){
                                new_mat.data[j * cols + k] -= mult * new_mat.data[r * cols + k];
                            }
                        }
                    }
                    lead++;
                }
                return new_mat;
            }
            Matrix augment(const Matrix& other){
                /*
                * Augment operator for Matrix class, returns the augmented matrix of two matrices
                * Basically, it just appends the other matrix to the right of this matrix
                * @throws invalid_argument if the matrices do not have the same number of rows
                * @param other: Matrix to be augmented with this matrix
                * @return: new Matrix that is the augmented matrix of this matrix and other
                */
                if (rows != other.rows){
                    throw std::invalid_argument("Matrices Must Have Same Number of Rows");
                }
                Matrix new_mat(rows, cols + other.cols);
                for (int row = 0; row < rows; row++){
                    for (int col = 0; col < cols; col++){
                        // fill in the new_mat with the data from the first matrix
                        new_mat.data[row * (cols + other.cols) + col] = data[row * cols + col];
                    }
                    for (int col = 0; col < other.cols; col++){
                        // fill in the new_mat with data from other
                        new_mat.data[row * (cols + other.cols) + cols + col] = other.data[row * other.cols + col];
                    }
                }
                return new_mat;
            }
            Matrix operator|(const Matrix& other){
                /*
                * Augment operator for Matrix class, returns the augmented matrix of two matrices
                * @param other: Matrix to be augmented with this matrix
                * @return: new Matrix that is the augmented matrix of this matrix and other
                */
                return augment(other);
            }
            Matrix& operator|= (const Matrix& other){
                /*
                * Augment assignment operator for Matrix class, concatenates other matrix to the right of this matrix
                * @param other: Matrix to be augmented with this matrix
                * @return: reference to this matrix
                */
                *this = augment(other);
                return *this;
            }
            Matrix identity(){
                /*
                * Identity operator for Matrix class, returns the identity matrix of the same size as this matrix
                * @throws invalid_argument if the matrix is not square
                * @return: new Matrix that is the identity matrix of the same size as this matrix
                */
                if(rows != cols){
                    throw std::invalid_argument("Matrix Must Be Square");
                }
                Matrix new_mat(rows, rows);
                for (int row = 0; row < rows; row++){
                    new_mat.data[row * rows + row] = 1;
                }
                return new_mat;
            }
            Matrix inverse(){
                /*
                * Inverse operator for Matrix class, returns the inverse of this matrix
                * @throws invalid_argument if the matrix is not square
                * @return: new Matrix that is the inverse of this matrix
                */
                if(rows != cols){
                    throw std::invalid_argument("Matrix Must Be Square");
                }
                if (det() == 0){
                    // if the determinant is 0, the matrix it is singular and has no inverse
                    throw std::invalid_argument("Matrix is Non-Invertible");
                }
                Matrix new_mat = augment(identity());
                new_mat = new_mat.rref();
                Matrix inverse_mat(rows, cols);
                for (int row = 0; row < rows; row++){
                    for (int col = 0; col < cols; col++){
                        inverse_mat.data[row * cols + col] = new_mat.data[row * (cols + rows) + cols + col];
                    }
                }
                return inverse_mat;
            }
            double& operator()(int row, int col){
                /*
                * Access operator for Matrix class, returns the value at the specified row and column
                * @throws invalid_argument if the row or column is out of bounds
                * @param row: row of the value to be accessed
                * @param col: column of the value to be accessed
                * @return: reference to the value at the specified row and column
                */
                if(row >= rows || col >= cols || row < 0 || col < 0){
                    throw std::out_of_range("Matrix Index Out of Bounds Exception");
                }
                return data[row*cols + col];
            }
            int getRows() const{
                /*
                * Getter for the number of rows in the matrix
                * @return: number of rows in the matrix
                */
                return rows;
            }
            int getCols() const{
                /*
                * Getter for the number of columns in the matrix
                * @return: number of columns in the matrix
                */
                return cols;
            }
            void print() const{
                /*
                * Prints the matrix to the console
                */
                std::cout.precision(4);
                std::cout << std::scientific;
                for (int row = 0; row < rows; row++){
                    std::cout << "[";
                    for (int col = 0; col < cols; col++){
                        std::cout << data[row*cols + col] << " ";
                    }
                    std::cout << "]"<< std::endl;
                }
            }
    
        private:
            int rows;
            int cols;
            double* data;
    };
    Matrix identity(int size){
        /*
        * Identity operator for Matrix class, returns the identity matrix of the specified size
        * @param size: size of the identity matrix
        * @return: new Matrix that is the identity matrix of the specified size
        */
        Matrix new_mat(size, size);
        for (int row = 0; row < size; row++){
            // set diagonal to 1
            new_mat(row, row) = 1;
        }
        return new_mat;
    }
}
#endif