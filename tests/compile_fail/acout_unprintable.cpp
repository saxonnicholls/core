//
//  acout_unprintable.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: streaming a type that nothing in the printing engine accepts
//  (no operator<<, no print member, not a container or tuple) into an acout. Built
//  with SNICHOLLS_COMPILE_FAIL it must fail to compile with the message below;
//  without it (the same type given a print member), it is the positive control and
//  must compile.
//

#include <snicholls/core/acout.hpp>

#include <sstream>

struct opaque {
    int value = 3;
#ifndef SNICHOLLS_COMPILE_FAIL
    void print(std::ostream& os) const { os << value; }
#endif
};

int main()
{
    std::ostringstream target;
    snicholls::acout(target) << opaque{};
    return target.str() == "3" ? 0 : 1;
}
