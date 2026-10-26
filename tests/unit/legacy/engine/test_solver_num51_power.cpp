/**
 * @file      test_solver_num51_power.cpp
 * @brief     Legacy-engine gate for NUM-51: `^` in [TREATMENT] expressions
 *            must never produce a non-finite value.
 *
 * @details Reproduces the swmm-code-review NUM-51 finding end-to-end on the
 *          legacy evaluator. Junction J1 receives 1 cfs DWF carrying
 *          TSS = 10 mg/L and the [TREATMENT] equations raise signed
 *          differences of TSS to powers:
 *
 *            TP   C = (TSS - 13)^2       = (-3)^2       = 9
 *            COD  C = 30 + (TSS - 12)^3  = 30 + (-2)^3  = 22
 *            BOD  C = 5 + (TSS - 14)^0.5 = 5 + (-4)^0.5 = 5
 *
 *          (-4)^0.5 has no real value; like sqrt() of a negative number it
 *          must evaluate to 0. The shipped legacy `r2 <= 0.0` guard in
 *          mathexpr.c collapses *every* negative-base power to 0 (so TP and
 *          COD read 0 and 30); the corrected rule keeps whole-number powers
 *          real and returns 0 for fractional exponents and a zero base.
 *
 * @author   Caleb Buahin <caleb.buahin@gmail.com>
 * @copyright Copyright (c) 2026 Caleb Buahin. OpenSWMM contributions only; original EPA SWMM
 *            credits are retained.
 * @license   Apache-2.0 for OpenSWMM contributions; original EPA SWMM provenance is retained.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "openswmm_solver.h"

namespace {

/**
 * @brief Write the NUM-51 power reproducer deck.
 *
 * Same deck as the review's NUM-51_power.inp: kinematic-wave routing of a
 * constant DWF so concentrations drain to the treatment equation values
 * within an hour.
 */
void write_deck(const std::string& tag) {
    std::ofstream f(tag + ".inp");
    f << "[TITLE]\n"
         "NUM-51: power of a non-positive base in [TREATMENT]\n\n"
         "[OPTIONS]\n"
         "FLOW_UNITS           CFS\n"
         "FLOW_ROUTING         KINWAVE\n"
         "START_DATE           01/01/2020\n"
         "START_TIME           00:00:00\n"
         "REPORT_START_DATE    01/01/2020\n"
         "REPORT_START_TIME    00:00:00\n"
         "END_DATE             01/01/2020\n"
         "END_TIME             01:00:00\n"
         "DRY_STEP             00:05:00\n"
         "WET_STEP             00:05:00\n"
         "ROUTING_STEP         30\n"
         "REPORT_STEP          00:15:00\n\n"
         "[JUNCTIONS]\n"
         ";;Name  Elev  MaxDepth  InitDepth  SurDepth  Aponded\n"
         "J1      2     5         0          0         0\n\n"
         "[OUTFALLS]\n"
         ";;Name  Elev  Type\n"
         "O1      0     FREE\n\n"
         "[CONDUITS]\n"
         ";;Name  From  To  Length  Roughness  InOffset  OutOffset  InitFlow\n"
         "C1      J1    O1  400     0.013      0         0          0\n\n"
         "[XSECTIONS]\n"
         ";;Link  Shape     Geom1  Geom2  Geom3  Geom4  Barrels\n"
         "C1      CIRCULAR  1      0      0      0      1\n\n"
         "[POLLUTANTS]\n"
         ";;Name  Units  Crain  Cgw  Crdii  Kdecay  SnowOnly  CoPollut  CoFrac  Cdwf  Cinit\n"
         "TSS     MG/L   0      0    0      0       NO        *         0       0     0\n"
         "TP      MG/L   0      0    0      0       NO        *         0       0     0\n"
         "COD     MG/L   0      0    0      0       NO        *         0       0     0\n"
         "BOD     MG/L   0      0    0      0       NO        *         0       0     0\n\n"
         "[DWF]\n"
         ";;Node  Constituent  Baseline\n"
         "J1      FLOW         1\n"
         "J1      TSS          10\n"
         "J1      TP           20\n"
         "J1      COD          40\n"
         "J1      BOD          20\n\n"
         "[TREATMENT]\n"
         ";;Node  Pollutant  Function\n"
         "J1      TP         C = (TSS - 13)^2\n"
         "J1      COD        C = 30 + (TSS - 12)^3\n"
         "J1      BOD        C = 5 + (TSS - 14)^0.5\n\n"
         "[REPORT]\n"
         "NODES ALL\n"
         "LINKS ALL\n";
}

/**
 * @brief Run the fixture and sample J1's concentrations for all four pollutants.
 *
 * Pollutant order follows the [POLLUTANTS] rows: TSS, TP, COD, BOD.
 */
struct Num51Run {
    bool ok = false;
    double conc[4] = {-1.0, -1.0, -1.0, -1.0};
};

Num51Run run_deck(const std::string& tag) {
    Num51Run r;
    write_deck(tag);
    if (swmm_open((tag + ".inp").c_str(), (tag + ".rpt").c_str(), (tag + ".out").c_str()) != 0) {
        ADD_FAILURE() << "swmm_open failed for " << tag;
        return r;
    }
    if (swmm_start(1) != 0) {
        ADD_FAILURE() << "swmm_start failed for " << tag;
        swmm_close();
        return r;
    }
    double elapsed = 0.0;
    int guard = 0;
    do {
        if (swmm_step(&elapsed) != 0) {
            ADD_FAILURE() << "swmm_step failed for " << tag;
            swmm_end();
            swmm_close();
            return r;
        }
    } while (elapsed > 0.0 && ++guard < 200000);

    const int j1 = swmm_getIndex(swmm_NODE, "J1");
    if (j1 < 0) {
        ADD_FAILURE() << "node lookup failed (j1=" << j1 << ")";
    } else {
        for (int k = 0; k < 4; ++k)
            r.conc[k] =
                swmm_getValueExpanded(swmm_NODE, swmm_NODE_POLLUTANT_CONCENTRATION, j1, 0, k);
        r.ok = true;
    }
    swmm_end();
    swmm_close();
    return r;
}

} // namespace

// Whole-number powers of a negative base are real; the legacy engine shipped
// with the NUM-51 defect reads 0.0 for TP and 30.0 for COD here.
TEST(LegacyNum51Power, WholeNumberExponentsMatchMath) {
    const Num51Run r = run_deck("_num51_power");
    ASSERT_TRUE(r.ok);
    EXPECT_NEAR(r.conc[1], 9.0, 0.1) << "TP: (TSS-13)^2 with TSS=10";
    EXPECT_NEAR(r.conc[2], 22.0, 0.1) << "COD: 30 + (TSS-12)^3 with TSS=10";
}

// A fractional power of a negative base has no real value and must be 0, as
// sqrt() of a negative number is in both engines; the shipped guard is also
// silent here, but the corrected rule must not produce NaN either.
TEST(LegacyNum51Power, FractionalExponentIsZeroNotNaN) {
    const Num51Run r = run_deck("_num51_power");
    ASSERT_TRUE(r.ok);
    EXPECT_NEAR(r.conc[3], 5.0, 0.1) << "BOD: 5 + (TSS-14)^0.5 with TSS=10";
    EXPECT_EQ(r.conc[3], r.conc[3]) << "BOD concentration must be finite, not NaN";
}
