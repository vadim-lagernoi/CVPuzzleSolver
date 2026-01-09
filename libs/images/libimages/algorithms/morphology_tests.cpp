#include "morphology.h"

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/algorithms/threshold_masking.h>
#include <libimages/tests_utils.h>

namespace fs = std::filesystem;

static image8u make_black(int w, int h) {
    image8u img(w, h, 1);
    img.fill(static_cast<std::uint8_t>(0));
    return img;
}

static void draw_filled_rect(image8u& img, int x0, int y0, int x1, int y1, std::uint8_t v) {
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i)
            img(j, i) = v;
}

static int count_white(const image8u& img) {
    int cnt = 0;
    for (int j = 0; j < img.height(); ++j)
        for (int i = 0; i < img.width(); ++i)
            if (img(j, i) == 255) ++cnt;
    return cnt;
}

TEST(morphology, SquareErodeDilate_R2) {
    configureWorkingDirectory();

    const fs::path dir = "debug-unit-tests/morphology/case00_square";
    fs::create_directories(dir);

    image8u in = make_black(32, 32);
    draw_filled_rect(in, 8, 8, 23, 23, 255);

    const auto er = morphology::erode(in, 2);
    const auto di = morphology::dilate(in, 2);


    debug_io::dump_image(getUnitCaseDebugDir() + "00_input.png", in);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_erode_r2.png", er);
    debug_io::dump_image(getUnitCaseDebugDir() + "02_dilate_r2.png", di);

    // Erosion shrinks: corner near original boundary becomes 0, deep inside stays 255.
    EXPECT_EQ(er(9, 9), 0);
    EXPECT_EQ(er(11, 11), 255);

    // Dilation expands: outside original square becomes 255 near it.
    EXPECT_EQ(di(6, 6), 255);
    EXPECT_EQ(di(5, 5), 0); // with r=2, (5,5) is distance 3 from (8,8) corner => stays 0
}

TEST(morphology, SinglePixel_ErodeAndDilate) {
    configureWorkingDirectory();

    const fs::path dir = "debug-unit-tests/morphology/case01_single_pixel";
    fs::create_directories(dir);

    image8u in = make_black(25, 25);
    in(12, 12) = 255;

    const auto er = morphology::erode(in, 1);
    const auto di = morphology::dilate(in, 2);

    debug_io::dump_image(getUnitCaseDebugDir() + "00_input.png", in);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_erode_r1.png", er);
    debug_io::dump_image(getUnitCaseDebugDir() + "02_dilate_r2.png", di);

    EXPECT_EQ(count_white(in), 1);
    EXPECT_EQ(count_white(er), 0);

    // Dilation with radius 2 -> (2r+1)^2 = 25 pixels
    EXPECT_EQ(count_white(di), 25);
    EXPECT_EQ(di(10, 10), 255);
    EXPECT_EQ(di(14, 14), 255);
    EXPECT_EQ(di(9, 9), 0);
}

TEST(morphology, StrengthZero_IsCopy) {
    configureWorkingDirectory();

    const fs::path dir = "debug-unit-tests/morphology/case02_strength0";
    fs::create_directories(dir);

    image8u in = make_black(16, 16);
    draw_filled_rect(in, 3, 5, 10, 12, 255);

    const auto er0 = morphology::erode(in, 0);
    const auto di0 = morphology::dilate(in, 0);

    debug_io::dump_image(getUnitCaseDebugDir() + "00_input.png", in);
    debug_io::dump_image(getUnitCaseDebugDir() + "01_erode_r0.png", er0);
    debug_io::dump_image(getUnitCaseDebugDir() + "02_dilate_r0.png", di0);

    EXPECT_EQ(count_white(er0), count_white(in));
    EXPECT_EQ(count_white(di0), count_white(in));
    EXPECT_EQ(er0(7, 7), in(7, 7));
    EXPECT_EQ(di0(7, 7), in(7, 7));
}

TEST(morphology, thresholdByConstant100AndUseMorphology) {
    configureWorkingDirectory();

    image8u img = load_image("data/00_photo_six_parts_downscaled_x4.jpg");
    image32f grayscale = to_grayscale_float(img);
    image8u is_foreground_mask = threshold_masking(grayscale, 100);
    debug_io::dump_image(getUnitCaseDebugDir() + "00_is_foreground_by_100.jpg", is_foreground_mask);

    int strength = 2;

    {
        image8u dilated = morphology::dilate(is_foreground_mask, strength);
        debug_io::dump_image(getUnitCaseDebugDir() + "11_dilated.jpg", dilated);

        image8u dilated_and_eroded = morphology::erode(dilated, strength);
        debug_io::dump_image(getUnitCaseDebugDir() + "12_dilated_and_eroded.jpg", dilated_and_eroded);
    }

    {
        image8u eroded = morphology::dilate(is_foreground_mask, strength);
        debug_io::dump_image(getUnitCaseDebugDir() + "21_eroded.jpg", eroded);

        image8u eroded_and_dilated = morphology::erode(eroded, strength);
        debug_io::dump_image(getUnitCaseDebugDir() + "22_eroded_and_dilated.jpg", eroded_and_dilated);
    }
}
