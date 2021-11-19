#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>
#include <cmath>
#include <iterator>

/*
* 这是一个base class即BaseMatrix
* BaseMatrix主要用于检查形式并构造矩阵
* BaseMatrix同时定义并实现Matrix类的迭代器
* 迭代器的category为random_access_iterator_tag
* 它的derived class是Matrix
*/
template<typename Data>
class BaseMatrix {
public:
    BaseMatrix &operator=(const BaseMatrix &);

protected:
    BaseMatrix(const std::initializer_list<std::initializer_list<Data>> &);

    BaseMatrix(const std::initializer_list<Data> &);

    explicit BaseMatrix(std::vector<std::vector<Data>> &);

    explicit BaseMatrix(std::vector<Data> &);

    explicit BaseMatrix(Data);

    BaseMatrix(const BaseMatrix &);

    BaseMatrix(std::size_t, std::size_t, Data);

    ~BaseMatrix() = default;

    std::vector<std::vector<Data>> array;
    std::size_t col_num;
    std::size_t row_num;
private:
    void init(std::vector<std::vector<Data>> &&); //用于copy和aopy-assignment函数调用
    bool check_vector(const std::vector<std::vector<Data>> &);
};

template<typename Data>
class Matrix;

template<typename Data>
std::ostream &operator<<(std::ostream &, Matrix<Data> &);

template<typename Data>
std::ostream &operator<<(std::ostream &, Matrix<Data> &&);//declaration

template<typename Data>
Matrix<Data> operator*(Data, Matrix<Data> &);

template<typename Data>
Matrix<Data> operator*(Data, Matrix<Data> &&);

template<typename Data>
Data det(Matrix<Data> &); //对Matrix求行列式
template<typename Data>
Data det(Matrix<Data> &&);

template<typename Data>
Matrix<Data> inv(Matrix<Data> &);//对Matrix求逆

template<typename Data>
Matrix<Data> inv(Matrix<Data> &&);

template<typename Data>
Data norm(Matrix<Data> &);//对Matrix求模

template<typename Data>
Data norm(Matrix<Data> &&);

template<typename Data>
class Matrix : public BaseMatrix<Data> {
    friend std::ostream &operator<<
    <Data>(std::ostream &, Matrix<Data> &);

    friend Data det<Data>(Matrix<Data> &);

    friend Data det<Data>(Matrix<Data> &&);

public:
    Matrix(const std::initializer_list<std::initializer_list<Data>> &list) : BaseMatrix<Data>(list) {}

    Matrix(const std::initializer_list<Data> &list) : BaseMatrix<Data>(list) {}

    explicit Matrix(std::vector<std::vector<Data>> &vvd) : BaseMatrix<Data>(vvd) {}

    explicit Matrix(std::vector<Data> &vd) : BaseMatrix<Data>(vd) {}

    explicit Matrix(Data d) : BaseMatrix<Data>(d) {}

    explicit Matrix(const BaseMatrix<Data> &bm) : BaseMatrix<Data>(bm) {} //BaseMatrix<Data>的引用可以动态绑定Matrix对象
    Matrix(std::size_t row, std::size_t col, Data d) : BaseMatrix<Data>(row, col, d) {}

    Matrix &operator=(const BaseMatrix<Data> &);

    Matrix &operator=(BaseMatrix<Data> &&);

    Matrix &operator=(const std::initializer_list<std::initializer_list<Data>> &);

    Matrix &operator=(const std::initializer_list<Data> &);

    Matrix operator-();    //矩阵的取反

    Matrix operator+(Matrix &);

    Matrix operator+(Matrix &&);

    Matrix operator-(Matrix &);

    Matrix operator-(Matrix &&);

    Matrix operator*(Matrix &);

    Matrix operator*(Matrix &&);

    Matrix operator*(Data);

    Matrix operator~();     //矩阵的转置

    Matrix operator!();     //矩阵求逆

    /*
     * 下述两个成员方法为模拟matlab执行切片功能
     * 该方法获得矩阵某一块的右值返回
     */
    Matrix operator()(const std::initializer_list<int> &, const std::initializer_list<int> &);

    Matrix operator()(const std::initializer_list<int> &);

    std::pair<std::size_t, std::size_t> get_size() const;

    std::vector<Data> &operator[](std::size_t i) { return this->array[i]; }

    Matrix slice(std::size_t, std::size_t = 0); //执行切片功能，为计算行列式det()的辅助函数，目前仅允许一次移除一行和一列
private:

    Data det(); //求矩阵的行列式
    /*
    *bool值用于检查加法或乘法的矩阵是否合法
    * 约定true为检查加法操作，false为乘法操作
    */
    bool check_matrix(const Matrix &, bool);

    void try_catch(const Matrix &, bool); //用于检查异常
};


/*
 * 重载vector的输出方式
 * 便于输出Matrix的行向量
 */
template<typename Data>
std::ostream &operator<<(std::ostream &, std::vector<Data> &);




/*
 * 以下为模板函数的实现代码
 */

/*
* 以下为实现函数时定义的临时函数的定义
* 这些临时函数的scope仅限于该文件
* 可忽略这些临时函数
*/


/*
 * 使用引用作为参数，会改变引用的值
 */
void operator_parentheses_set_index(size_t &begin_index, size_t &end_index, int &spacing,
                                    const std::initializer_list<int> &list) {
    switch (list.size()) {
        case (3):
            spacing = *(list.begin() + 1); //无break，故会继续case(2)语句，为begin_index和end_index赋值
        case (2):
            begin_index = static_cast<size_t>(*(list.begin()) - 1);
            end_index = static_cast<size_t>(*(list.end() - 1));
            end_index += (spacing > 0) ? 0 : -1;
            break;
        case (1):
            begin_index = end_index = static_cast<size_t>(*(list.begin()) - 1);
            ++end_index;
            break;
        default:
            throw std::runtime_error("more parameters than requirements");
    }
}

template<typename Data>
std::ostream &operator<<(std::ostream &os, std::vector<Data> &vd) {
    for (auto iter = vd.cbegin(); iter != vd.cend(); ++iter) {
        os << *iter << "\t";
    };
    return os;
}


/*
* 本实现文件中，由于BaseMatrix<Data>::init函数传入的参数需调用swap()函数，
* 且参数仅用于初始化，故传入右值引用。
* 为增强代码robustness，intalizer、copy、copy-assignment函数采用左值引用传参，
* 再通过std::move()函数转为右值引用
*/

template<typename Data>
bool BaseMatrix<Data>::check_vector(const std::vector<std::vector<Data>> &vvd) {
    size_t length = vvd[0].size();
    for (auto vd: vvd) {
        if (length != vd.size()) {
            throw std::invalid_argument("each row requires the same number of elements");
            return false;
        }
    }
    return true;
}

template<typename Data>
void BaseMatrix<Data>::init(std::vector<std::vector<Data>> &&vvd) {
    try { //检查参数是否满足矩阵要求，若不满足则结束进程
        check_vector(vvd);
    }
    catch (std::invalid_argument &err) {
        std::cerr << err.what() << std::endl;
        std::abort();
    }
    row_num = vvd.size();
    col_num = vvd[0].size();
    array.swap(vvd);
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(std::vector<std::vector<Data>> &vvd) :
        col_num(0), row_num(0), array() {
    init(std::vector<std::vector<Data>>(vvd));
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(Data d) :
        col_num(0), row_num(0), array() {
    std::vector<std::vector<Data>> vvd;
    std::vector<Data> vd(d);
    vd.push_back(d);
    vvd.push_back(vd);
    init(std::vector<std::vector<Data>>(vvd));
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(const BaseMatrix<Data> &rhs) :
        col_num(0), row_num(0), array() {
    init(std::vector<std::vector<Data>>(rhs.array));
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(const std::initializer_list<std::initializer_list<Data>> &list):
        col_num(0), row_num(0), array() {
    std::vector<std::vector<Data>> vvd;
    for (auto &list_vd: list) {
        std::vector<Data> vd;
        for (auto &list_d: list_vd)
            vd.push_back(list_d);
        vvd.push_back(vd);
    }
    init(std::move(vvd));
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(const std::initializer_list<Data> &list):
        col_num(0), row_num(0), array() {
    std::vector<std::vector<Data>> vvd;
    std::vector<Data> vd;
    for (auto list_iter = list.begin(); list_iter != list.end(); ++list_iter) {
        vd.push_back(*list_iter);
    }
    vvd.push_back(vd);
    init(std::move(vvd));
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(std::vector<Data> &rhs) :
        col_num(0), row_num(0), array() {
    std::vector<std::vector<Data>> vvd;
    vvd.push_back(rhs);
    init(std::move(vvd));
}

template<typename Data>
BaseMatrix<Data>::BaseMatrix(size_t row, size_t col, Data d):
        col_num(0), row_num(0), array() {
    std::vector<std::vector<Data>> vvd;
    for (size_t i = 0; i != row; ++i) {
        std::vector<Data> vd(col, d);
        vvd.push_back(vd);
    }
    init(std::move(vvd));
}

template<typename Data>
BaseMatrix<Data> &BaseMatrix<Data>::operator=(const BaseMatrix<Data> &rhs) {
    init(std::vector<std::vector<Data>>(rhs.array));
    return *this;
}


/*
* 以下为Matrix<Data>类的定义
*/
template<typename Data>
bool Matrix<Data>::check_matrix(const Matrix<Data> &rhs, bool judge) {
    if (judge) {
        if (!(this->col_num == rhs.col_num && this->row_num == rhs.row_num)) {
            throw std::invalid_argument("both matrices require the same number of row_num and col_num");
            return false;
        }
        return true;
    } else {
        if (this->col_num != rhs.row_num && this->row_num != rhs.col_num) {
            throw std::invalid_argument("the lhs' col_num should be the same with rhs', and vice versa");
        }
        return true;
    }
}

template<typename Data>
void Matrix<Data>::try_catch(const Matrix<Data> &rhs, bool judge) {
    try { check_matrix(rhs, judge); }
    catch (std::invalid_argument &err) {
        std::cerr << err.what() << std::endl;
        std::abort();
    }
}

template<typename Data>
Matrix<Data> &Matrix<Data>::operator=(const BaseMatrix<Data> &bm) {
    BaseMatrix<Data> &temp = *this;
    temp = bm;
    return *this;
}

template<typename Data>
Matrix<Data> &Matrix<Data>::operator=(BaseMatrix<Data> &&bm) {
    if (this == &bm) return *this;
    *this = std::move(bm);
    return *this;
}

template<typename Data>
Matrix<Data> &Matrix<Data>::operator=(const std::initializer_list<std::initializer_list<Data>> &list) {
    Matrix<Data> temp(list);
    *this = temp;
    return *this;
}

template<typename Data>
Matrix<Data> &Matrix<Data>::operator=(const std::initializer_list<Data> &list) {
    Matrix<Data> temp(list);
    *this = temp;
    return *this;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator-() {
    Matrix<Data> temp(*this);
    for (size_t i = 0; i != temp.row_num; ++i) {
        for (size_t j = 0; j != temp.col_num; ++j) temp[i][j] = -temp[i][j];
    }
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator+(Matrix<Data> &rhs) {
    try_catch(rhs, true);
    Matrix<Data> temp(*this);
    for (size_t i = 0; i != temp.row_num; ++i) {
        for (size_t j = 0; j != temp.col_num; ++j) temp[i][j] += rhs[i][j];
    }
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator+(Matrix<Data> &&rhs) {
    Matrix<Data> temp = std::move(rhs);
    return *this + temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator-(Matrix<Data> &rhs) {
    try_catch(rhs, true);
    Matrix<Data> temp(*this);
    for (size_t i = 0; i != this->row_num; ++i) {
        for (size_t j = 0; j != this->col_num; ++j) temp.array[i][j] -= rhs[i][j];
    }
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator-(Matrix<Data> &&rhs) {
    Matrix<Data> temp = std::move(rhs);
    return *this - temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator~() {
    Matrix<Data> temp(*this);
    std::vector<std::vector<Data>> vvd;
    for (size_t i = 0; i != this->col_num; ++i) {
        std::vector<Data> vd;
        for (size_t j = 0; j != this->row_num; ++j) vd.push_back(this->array[j][i]);
        vvd.push_back(vd);
    }
    temp.array.swap(vvd);
    //转置完毕后交换row_num和col_num的值
    size_t t = temp.row_num;
    temp.row_num = temp.col_num;
    temp.col_num = t;
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator*(Matrix<Data> &rhs) {
    if (rhs.col_num == 1 && rhs.row_num == 1) return *this * rhs[0][0];
    else if (this->col_num == 1 && this->row_num == 1) return rhs * (*this)[0][0];
    else {
        try_catch(rhs, false);
        std::vector<std::vector<Data>> vvd;
        /*
        * 下面是乘法运算的逻辑部分
        * 然而这段代码连用了三个循环，且逻辑写得过于复杂，不漂亮
        * 在思考能否重写
        */
        for (size_t i = 0; i != this->row_num; ++i) {
            std::vector<Data> vd;
            for (size_t j = 0; j != rhs.col_num; ++j) { //由于rhs已转置，故取row_num，即其原来的col_num
                vd.push_back(0);
                for (size_t k = 0; k != this->col_num; ++k) vd[j] += (*this)[i][k] * rhs[k][j];
            }
            vvd.push_back(vd);
        }
        Matrix<Data> temp(vvd);
        //改变矩阵size的值
        temp.col_num = rhs.col_num;
        temp.row_num = this->row_num;
        return temp;
    }
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator*(Matrix<Data> &&rhs) {
    Matrix<Data> temp = std::move(rhs);
    return (*this) * temp;
}

template<typename Data>
std::pair<size_t, size_t> Matrix<Data>::get_size() const {
    return std::pair<size_t, size_t>(this->row_num, this->col_num);
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator*(Data d) {
    Matrix<Data> temp(*this);
    for (size_t i = 0; i != temp.row_num; ++i) {
        for (size_t j = 0; j != temp.col_num; ++j) temp[i][j] *= d;
    }
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::slice(std::size_t row, std::size_t col) {
    if (this->row_num == 1 && this->col_num == 1)
        throw std::invalid_argument("Matrix<Data>'s size can't be smaller than one");
    Matrix<Data> temp(*this);
    if (row <= 0);
    else if (row > temp.row_num) throw std::invalid_argument("The row index crossing the line");
    else {
        --row;
        temp.array.erase(temp.array.begin() + row);
        --temp.row_num;
    }
    if (col <= 0);
    else if (row > temp.col_num) throw std::invalid_argument("The column index crossing the line");
    else {
        --col;
        for (std::vector<Data> &vd: temp.array) {
            vd.erase(vd.begin() + col);
        }
        --temp.col_num;
    }
    return temp;
}

template<typename Data>
Data Matrix<Data>::det() {
    if (this->row_num != this->col_num)
        throw std::invalid_argument("Calculating determinant requirs an n-dimensional Matrix<Data>");
    if (this->row_num == 1 && this->col_num == 1) return this->array[0][0];
    else if (this->row_num == 2 && this->col_num == 2)
        return (this->array[0][0] * this->array[1][1] - this->array[0][1] * this->array[1][0]);
    else if (this->row_num == 3 && this->col_num == 3)
        return
                (this->array[0][0] * this->array[1][1] * this->array[2][2] +
                 this->array[1][0] * this->array[2][1] * this->array[0][2] +
                 this->array[0][1] * this->array[1][2] * this->array[2][0]
                 - this->array[0][2] * this->array[1][1] * this->array[2][0] -
                 this->array[0][1] * this->array[1][0] * this->array[2][2] -
                 this->array[0][0] * this->array[1][2] * this->array[2][1]);
    else {
        Data determinant = 0;
        for (size_t i = 0; i != this->row_num; ++i) {
            if (this->array[i][0] != 0) determinant += this->array[i][0] * pow(-1, i + 0) * slice(i + 1, 1).det();
        }
        return determinant;
    }
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator!() {
    if (det() == 0)
        throw std::runtime_error(
                "Matrix<Data>'s determinant is not equal to zero, which means it is not an invertible Matrix<Data>");
    if (this->row_num == 1 && this->col_num == 1) {
        this->array[0][0] = 1.0 / this->array[0][0];
        return *this;
    }
    Matrix<Data> temp(*this), slice_temp(*this);
    Data d;
    Data determinant = temp.det();
    for (size_t i = 0; i != temp.row_num; ++i) {
        for (size_t j = 0; j != temp.col_num; ++j) {
            d = pow(-1, i + j) * slice_temp.slice(i + 1, j + 1).det();
            temp[i][j] = (d == -0) ? 0 : d; //避免出现-0现象
        }
    }
    temp = ~temp;
    temp = temp * (1.0 / determinant);
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator()(const std::initializer_list<int> &row, const std::initializer_list<int> &col) {
    std::vector<std::vector<Data>> vvd;
    size_t col_begin_index, col_end_index;
    size_t row_begin_index, row_end_index;
    int col_spacing = 1, row_spacing = 1;
    operator_parentheses_set_index(col_begin_index, col_end_index, col_spacing, col);
    operator_parentheses_set_index(row_begin_index, row_end_index, row_spacing, row);
    /*
    * 使用三则运算符是为了保证i下标能够考虑到row_spacing
    */
    for (size_t i = row_begin_index; (row_spacing > 0) ? i < row_end_index : i > row_end_index; i += row_spacing) {
        std::vector<Data> vd;
        for (size_t j = col_begin_index; (col_spacing > 0) ? j < col_end_index : j > col_end_index; j += col_spacing) {
            vd.push_back(this->array[i][j]);
        }
        vvd.push_back(vd);
    }
    Matrix<Data> temp(vvd);
    return temp;
}

template<typename Data>
Matrix<Data> Matrix<Data>::operator()(const std::initializer_list<int> &col) {
    return (*this)({1}, col);
}

/*
* 以下为class scope外的重载函数定义
*/
template<typename Data>
std::ostream &operator<<(std::ostream &os, Matrix<Data> &rhs) {
    auto size = rhs.get_size();
    size_t row_num = size.first, col_num = size.second;
    os << "size: " << row_num << " * " << col_num << std::endl;
    for (size_t i = 0; i != row_num; ++i) {
        for (size_t j = 0; j != col_num; ++j) os << rhs[i][j] << " ";
        os << std::endl;
    }
    return os;
}

template<typename Data>
std::ostream &operator<<(std::ostream &os, Matrix<Data> &&rhs) {
    Matrix<Data> temp(rhs);
    return os << temp;
}

template<typename Data>
Matrix<Data> operator*(Data d, Matrix<Data> &rhs) {
    return rhs * d;
}

template<typename Data>
Matrix<Data> operator*(Data d, Matrix<Data> &&rhs) {
    return rhs * d;
}

template<typename Data>
Data det(Matrix<Data> &m) {
    return m.det();
}

template<typename Data>
Data det(Matrix<Data> &&m) {
    return m.det();
}

template<typename Data>
Matrix<Data> inv(Matrix<Data> &m) {
    return !m;
}

template<typename Data>
Matrix<Data> inv(Matrix<Data> &&m) {
    return !m;
}

template<typename Data>
Data norm(Matrix<Data> &m) {
    Data sum = 0;
    for (size_t i = 0; i != m.get_size().first; ++i) {
        for (size_t j = 0; j != m.get_size().second; ++j) sum += m[i][j];
    }
    return sqrt(sum);
}

template<typename Data>
Data norm(Matrix<Data> &&m) {
    Matrix<Data> temp(m);
    return norm(temp);
}