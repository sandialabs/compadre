#include "Compadre_Misc.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace Compadre;

class XyzTest: public ::testing::Test {
public:
    XYZ a;
    XYZ b;

    XyzTest( ) {
        // initialization
    }

    void SetUp( ) {
        // before test
        a = XYZ(1,1,1);
        b = XYZ(2,2,2);
    }

    void TearDown( ) {
        // after test completes
    }
};

TEST_F (XyzTest, XYZMultL) {
    ASSERT_DOUBLE_EQ (3.0, (a*3.0).x);
    ASSERT_DOUBLE_EQ (4.0, (a*4.0).y);
    ASSERT_DOUBLE_EQ (5.0, (a*5.0).y);
}
TEST_F (XyzTest, XYZMultR) {
    ASSERT_DOUBLE_EQ (3.0, (3.0*a).z);
}

int main(int argc, char **argv) {
        ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
