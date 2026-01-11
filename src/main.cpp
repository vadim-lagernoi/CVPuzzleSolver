#include <libimages/draw.h>
#include <libimages/algorithms/grayscale.h>
#include <libimages/algorithms/threshold_masking.h>
#include <libimages/algorithms/morphology.h>
#include <libimages/algorithms/split_into_parts.h>
#include <libimages/algorithms/extract_contour.h>
#include <libimages/algorithms/simplify_contours.h>

#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libbase/runtime_assert.h>
#include <libbase/configure_working_directory.h>
#include <libimages/debug_io.h>
#include <libimages/image.h>
#include <libimages/image_io.h>

#include <iostream>

int main() {
    try {
        configureWorkingDirectory();

        // это список картинок которые вы хотите обработать при запуске
        // сначала тестироваться лучше всего на маленькой картинке (первая в списке)
        // но если все работает и хочется дополнительно проверить алгоритм,
        // то раскомментируйте остальные строчки и проверьте алгоритм и на них
        // отладочная визуализация теперь сохраняется не напрямую в debug,
        // а в подпапке вида debug/00_photo_six_parts_downscaled_x4 (название соответствует картинке)
        std::vector<std::string> to_process = {
            "00_photo_six_parts_downscaled_x4",
            // "00_photo_six_parts",
            // "01_eight_parts",
            // "02_eight_parts_shuffled",
            // "03_eight_parts_shuffled2",
        };

        Timer all_images_t;
        for (const std::string &image_name: to_process) {
            Timer total_t;
            Timer t;

            std::string debug_dir = "debug/" + image_name + "/";
            image8u image = load_image("data/" + image_name + ".jpg");
            auto [w, h, c] = image.size();
            rassert(c == 3, 237045347618912, image.channels());
            std::cout << "image loaded in " << t.elapsed() << " sec" << std::endl;
            debug_io::dump_image(debug_dir + "00_input.jpg", image);

            image32f grayscale = to_grayscale_float(image);
            rassert(grayscale.channels() == 1, 2317812937193);
            rassert(grayscale.width() == w && grayscale.height() == h, 7892137419283791);
            debug_io::dump_image(debug_dir + "01_grayscale.jpg", grayscale);

            std::vector<float> intensities_on_border;
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    // пропускаем все пиксели кроме границы изображения
                    if (i != 0 && i != w - 1 && j != 0 && j != h - 1)
                        continue;
                    intensities_on_border.push_back(grayscale(j, i));
                }
            }
            // DONE: какой инвариант мы можем проверить про размер intensities_on_border.size()? чем он должен быть равен?
            rassert(intensities_on_border.size() == 2 * w + 2 * h - 4, 7283197129381312);
            std::cout << "intensities on border: " << stats::summaryStats(intensities_on_border) << std::endl;

            // DONE: найдем порог разделяющий яркость на фон и объект - background_threshold
            double background_threshold = 1.5 * stats::percentile(intensities_on_border, 90);
            std::cout << "background threshold=" << background_threshold << std::endl;

            // DONE: построим маску объект-фон + сохраним визуализацию на диск + выведем в лог процент пикселей на фоне
            image8u is_foreground_mask = threshold_masking(grayscale, background_threshold);
            double is_foreground_sum = stats::sum(is_foreground_mask.toVector());
            std::cout << "thresholded background: " << stats::toPercent(w * h - is_foreground_sum / 255.0, 1.0 * w * h) << std::endl;
            debug_io::dump_image(debug_dir + "02_is_foreground_mask.png", is_foreground_mask);

            t.restart();
            // DONE: сделаем маску более гладкой и точной через Морфологию
            // DONE: сначала попробуем dilation + erosion, все ли хорошо поулчилось? нет ли выбросов?
            int strength = 3;

            const bool with_openmp = true;
            image8u dilated_mask = morphology::dilate(is_foreground_mask, strength, with_openmp);
            image8u dilated_eroded_mask = morphology::erode(dilated_mask, strength, with_openmp);
            image8u dilated_eroded_eroded_mask = morphology::erode(dilated_eroded_mask, strength, with_openmp);
            image8u dilated_eroded_eroded_dilated_mask = morphology::dilate(dilated_eroded_eroded_mask, strength, with_openmp);
            std::cout << "full morphology in " << t.elapsed() << " sec" << std::endl;

            debug_io::dump_image(debug_dir + "03_is_foreground_dilated.png", dilated_mask);
            debug_io::dump_image(debug_dir + "04_is_foreground_dilated_eroded.png", dilated_eroded_mask);
            debug_io::dump_image(debug_dir + "05_is_foreground_dilated_eroded_eroded.png", dilated_eroded_eroded_mask);
            debug_io::dump_image(debug_dir + "06_is_foreground_dilated_eroded_eroded_dilated.png", dilated_eroded_eroded_dilated_mask);

            is_foreground_mask = dilated_eroded_eroded_dilated_mask;
            auto [objOffsets, objImages, objMasks] = splitObjects(image, is_foreground_mask);
            int objects_count = objImages.size();
            std::cout << objects_count << " objects extracted" << std::endl;
            rassert(objects_count == 6 || objects_count == 8, 237189371298, objects_count);

            // визуализируем цветами компоненты связности - один объект - один цвет
            image32i image_with_object_indices(image.width(), image.height(), 1);
            for (int obj = 0; obj < objects_count; ++obj) {
                // это отступ - координата верхнего левого угла объекта на оригинальной картинке
                point2i offset = objOffsets[obj];

                // это маска объекта
                image8u mask = objMasks[obj];

                for (int j = 0; j < mask.height(); ++j) {
                    for (int i = 0; i < mask.width(); ++i) {
                        // если объект в своей маске отмечен как "тут объект"
                        if (mask(j, i) == 255) {
                            // то рассчитываем координаты этого пикселя в оригинальной картинке и пишем туда наш номер (индексация с 1)
                            int global_i = offset.x + i;
                            int global_j = offset.y + j;
                            image_with_object_indices(global_j, global_i) = obj + 1;
                        }
                    }
                }
            }
            debug_io::dump_image(debug_dir + "07_colorized_objects.jpg", debug_io::colorize_labels(image_with_object_indices, 0));

            for (int obj = 0; obj < objects_count; ++obj) {
                std::string obj_debug_dir = debug_dir + "objects/object" + std::to_string(obj) + "/";

                debug_io::dump_image(obj_debug_dir + "01_image.jpg", objImages[obj]);
                debug_io::dump_image(obj_debug_dir + "02_mask.jpg", objMasks[obj]);

                // DONE реализуйте построение маски контура-периметра, нажмите Ctrl+Click на buildContourMask:
                image8u objContourMask = buildContourMask(objMasks[obj]);

                debug_io::dump_image(obj_debug_dir + "03_mask_contour.jpg", objContourMask);

                std::vector<point2i> contour = extractContour(objContourMask);

                // сделаем черную картинку чтобы визуализировать контур на ней
                image32f contour_visualization(objImages[obj].width(), objImages[obj].height(), 1);

                // нарисуем на ней контур
                for (int i = 0; i < contour.size(); ++i) {
                    point2i pixel = contour[i];
                    // сделаем цвет тем ярче - чем дальше пиксель в контуре (чтобы проверить что он по часовой стрелке)
                    drawPoint(contour_visualization, pixel, color32f(i * 255.0f / contour.size()));
                }

                debug_io::dump_image(obj_debug_dir + "04_mask_contour_clockwise.jpg", contour_visualization);

                // у нас теперь есть перечень пикселей на контуре объекта
                // DONE реализуйте определение в этом контуре 4 вершин-углов и нарисуйте их на картинке, нажмите Ctrl+Click на simplifyContour:
                std::vector<point2i> corners = simplifyContour(contour, 4);
                rassert(corners.size() == 4, 32174819274812);

                // сделаем черную картинку чтобы визуализировать вершины-углы на ней
                image32f corners_visualization(objImages[obj].width(), objImages[obj].height(), 1);
                for (point2i corner: corners) {
                    drawPoint(corners_visualization, corner, color32f(255.0f), 10);
                }
                debug_io::dump_image(obj_debug_dir + "05_corners_visualization.jpg", corners_visualization);

                // теперь извлечем стороны объекта
                std::vector<std::vector<point2i>> sides = splitContourByCorners(contour, corners);
                rassert(sides.size() == 4, 237897832141);

                // визуализируем каждую сторону объекта отдельным цветом:
                image8u sides_visualization(objImages[obj].width(), objImages[obj].height(), 3);
                FastRandom r(2391);
                for (int i = 0; i < sides.size(); ++i) {
                    color8u random_color = {(uint8_t) r.nextInt(0, 255), (uint8_t) r.nextInt(0, 255), (uint8_t) r.nextInt(0, 255)};
                    color8u side_color = random_color;
                    drawPoints(sides_visualization, sides[i], side_color);
                }
                debug_io::dump_image(obj_debug_dir + "06_sides.jpg", sides_visualization);
            }

            std::cout << "image " << image_name << " processed in " << total_t.elapsed() << " sec" << std::endl;
        }
        std::cout << "all images processed in " << all_images_t.elapsed() << " sec" << std::endl;

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
