#include <cstddef>
#include <iostream>

// arbitrary funciton f(x) = (x-a)^2
float func(const float x, const float a){
    return (x - a)*(x - a);
}


// derivative of the arbitrary function f(x)' = 2(x-a)
float deriv(const float x, const float a){
    return 2 * (x - a); 
}


int main(){

    float x {10.0f};
    constexpr float a {3.0f};
    constexpr float alpha {0.2f};
    
    for (std::size_t i{0}; i < 30; ++i){
        float fx {func(x, a)};
        float dfx {deriv(x, a)};

        std::cout <<"i: " << i << " " << "x: " << x << " " << "fx: " << fx << " " << "dfx: " << dfx << '\n';

        x -= alpha * dfx;
    }

    return 0;
}