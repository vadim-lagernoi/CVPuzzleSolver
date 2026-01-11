#pragma once

#include <libimages/image.h>
#include <libbase/point2.h>


std::tuple<std::vector<point2i>, std::vector<image8u>, std::vector<image8u>> splitObjects(
    const image8u &image, const image8u &objectsMask);
