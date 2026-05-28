/*
 *   test_gage_scalefactor.cpp
 *
 *   Created: 2026
 *
 *   Unit tests for rain gage scale factor feature (Google Test).
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "openswmm_solver.h"

// Paths relative to the test working directory (tests/unit/legacy/engine/data)
#define DATA_PATH_SCALE   "./gage_scalefactor/gage_scalefactor.inp"
#define DATA_PATH_NOSCALE "./gage_scalefactor/gage_noscalefactor.inp"
#define RPT_SCALE         "./gage_scalefactor/_scale_test.rpt"
#define OUT_SCALE         "./gage_scalefactor/_scale_test.out"
#define RPT_NOSCALE       "./gage_scalefactor/_noscale_test.rpt"
#define OUT_NOSCALE       "./gage_scalefactor/_noscale_test.out"

class GageScaleFactorTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        std::remove(RPT_SCALE);
        std::remove(OUT_SCALE);
        std::remove(RPT_NOSCALE);
        std::remove(OUT_NOSCALE);
    }
};

// Test that a model with no scale factor (backward compatible) runs successfully
TEST_F(GageScaleFactorTest, NoScaleFactorBackwardCompat)
{
    ASSERT_EQ(swmm_open(DATA_PATH_NOSCALE, RPT_NOSCALE, OUT_NOSCALE), 0);
    ASSERT_EQ(swmm_start(1), 0);

    double elapsedTime;
    int err;
    do {
        err = swmm_step(&elapsedTime);
        ASSERT_EQ(err, 0);
    } while (elapsedTime > 0.0);

    ASSERT_EQ(swmm_end(), 0);
    swmm_close();
}

// Test that a model with a scale factor of 2.0 runs successfully
TEST_F(GageScaleFactorTest, WithScaleFactorRuns)
{
    ASSERT_EQ(swmm_open(DATA_PATH_SCALE, RPT_SCALE, OUT_SCALE), 0);
    ASSERT_EQ(swmm_start(1), 0);

    double elapsedTime;
    int err;
    do {
        err = swmm_step(&elapsedTime);
        ASSERT_EQ(err, 0);
    } while (elapsedTime > 0.0);

    ASSERT_EQ(swmm_end(), 0);
    swmm_close();
}

// Test that scale factor of 2.0 produces doubled rainfall intensity
TEST_F(GageScaleFactorTest, ScaleFactorDoublesRainfall)
{
    double rainfall_noscale = 0.0;
    double rainfall_scale = 0.0;
    double elapsedTime;
    int err;

    // Run without scale factor and capture first nonzero rainfall
    ASSERT_EQ(swmm_open(DATA_PATH_NOSCALE, RPT_NOSCALE, OUT_NOSCALE), 0);
    ASSERT_EQ(swmm_start(1), 0);

    do {
        err = swmm_step(&elapsedTime);
        ASSERT_EQ(err, 0);
        if (elapsedTime > 0.0)
        {
            double r = swmm_getValue(swmm_GAGE_RAINFALL, 0);
            if (r > 0.0 && rainfall_noscale == 0.0)
                rainfall_noscale = r;
        }
    } while (elapsedTime > 0.0);
    swmm_end();
    swmm_close();

    // Run with scale factor = 2.0 and capture first nonzero rainfall
    ASSERT_EQ(swmm_open(DATA_PATH_SCALE, RPT_SCALE, OUT_SCALE), 0);
    ASSERT_EQ(swmm_start(1), 0);

    do {
        err = swmm_step(&elapsedTime);
        ASSERT_EQ(err, 0);
        if (elapsedTime > 0.0)
        {
            double r = swmm_getValue(swmm_GAGE_RAINFALL, 0);
            if (r > 0.0 && rainfall_scale == 0.0)
                rainfall_scale = r;
        }
    } while (elapsedTime > 0.0);
    swmm_end();
    swmm_close();

    // The scaled rainfall should be 2x the unscaled
    ASSERT_GT(rainfall_noscale, 0.0);
    ASSERT_GT(rainfall_scale, 0.0);
    EXPECT_NEAR(rainfall_scale, 2.0 * rainfall_noscale,
                rainfall_noscale * 0.0001);
}
