#include <gtest/gtest.h>

#include <gpu_planning/block_buffer.hpp>

TEST(GTestTest, WorkingTest) { EXPECT_EQ(true, 1 > 0) << "Working test"; }

TEST(GTestTest, FailingTest) { EXPECT_EQ(true, 2 > 3) << "Failing test"; }
