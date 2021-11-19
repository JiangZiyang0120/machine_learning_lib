#include <iostream>
#include <random>
#include <map>
#include <string>
#include <tuple>
#include "division.h"
#include "measurement.h"
#include "matrix.h"
#include "util_algorithm.h"
#include "linear_model.h"

using namespace std;

int main(){
    Matrix a = {{1.2, 3.4, 5.6},
                {2.1,2.3,3.3},
                {1.2,2.0,2.7}};
    cout<<inv(a*~a)<<endl;
}