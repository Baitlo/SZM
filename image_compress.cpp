#include "image_compress.hpp"

#include "k_means.hpp"

#include "glog/logging.h"
#include "common/utils.hpp"

#include "nlohmann/json.hpp"
#include "tiffio.h"
#include <cstddef>
#include <cstring>

#include <fstream>
#include <vector>

typedef uint8_t RGB_BYTETYPE;
typedef uint8_t YCRCB_BYTETYPE;

static std::filesystem::path debug_output_file_path = "/tmp/compress/debug.out";
static std::ofstream debug_output_file(debug_output_file_path);

ImageCompressor::ImageCompressor(const char* input_file_name) {
    // base_path: folder
    // input_file_name: input tiff file name
    // compres_output_file_path: the byte array gotten by compressing
    // restore_output_file_path: the new tiff file restored from the compressed byte array
    // text_output_file_path: for a width * height image, for filter-white points, K clusters points, 
    //                        use '*', '+', '_' and so on to represent each cluster and output as
    //                        txt file. This can explicitly shows how the image looks like.
    input_file_path = base_path / std::string(input_file_name);
    compres_output_file_path = base_path / ("compress_" + std::string(input_file_name) + ".bin");
    restore_output_file_path = base_path / ("restore_" + std::string(input_file_name));
    text_output_file_path = base_path / ("distribution_" + std::string(input_file_name) + ".txt");

    LOG(INFO) << "input_file_name=" << input_file_path << ", compres_output_file_path="
              << compres_output_file_path << ", restore_output_file_path="
              << restore_output_file_path;
}

static YCRCB_BYTETYPE get_y_from_RGB(RGB_BYTETYPE r, RGB_BYTETYPE g, RGB_BYTETYPE b) {
    return (YCRCB_BYTETYPE) (int) (0.257 * r + 0.504 * g + 0.098 * b + 16);
}

static YCRCB_BYTETYPE get_cb_from_RGB(RGB_BYTETYPE r, RGB_BYTETYPE g, RGB_BYTETYPE b) {
    return (YCRCB_BYTETYPE) (int) (-0.148 * r - 0.291 * g + 0.439 * b + 128);
}

static YCRCB_BYTETYPE get_cr_from_RGB(RGB_BYTETYPE r, RGB_BYTETYPE g, RGB_BYTETYPE b) {
    return (YCRCB_BYTETYPE) (int) (0.439 * r - 0.368 * g - 0.071 * b + 128);
}

static RGB_BYTETYPE get_r_from_YCBCR(YCRCB_BYTETYPE y, YCRCB_BYTETYPE cb, YCRCB_BYTETYPE cr) {
    return (RGB_BYTETYPE) (int) (1.164 * (y - 16) + 1.596 * (cr - 128));
}

static RGB_BYTETYPE get_g_from_YCBCR(YCRCB_BYTETYPE y, YCRCB_BYTETYPE cb, YCRCB_BYTETYPE cr) {
    return (RGB_BYTETYPE) (int) (1.164 * (y - 16) - 0.392 * (cb - 128) - 0.813 * (cr - 128));
}

static RGB_BYTETYPE get_b_from_YCBCR(YCRCB_BYTETYPE y, YCRCB_BYTETYPE cb, YCRCB_BYTETYPE cr) {
    return (RGB_BYTETYPE) (int) (1.164 * (y - 16) + 2.017 * (cb - 128));
}

static void encode_white_pixels(std::ofstream& ostream, uint8_t count) {
	uint8_t v = 3; // 0x11 bits
	v = v << 6;
	v = v | (count & 0xFF);
    ostream.write(reinterpret_cast<const char*>(&v), 1);

    debug_output_file << (int) count << " ";
}

static void encode_cluster_pixels(std::ofstream& ostream, int cluster_center_index,
                                  std::vector<std::vector<double>>& clustering_center_points,
                                  std::vector<double>& one_ycbcr,
                                  int multiplier) {
    YCRCB_BYTETYPE base_cb = clustering_center_points.at(cluster_center_index).at(0);
    YCRCB_BYTETYPE base_cr = clustering_center_points.at(cluster_center_index).at(1);
    // LOG(INFO) << "pass clustering_center_points";

    YCRCB_BYTETYPE local_y = one_ycbcr.at(0);
    YCRCB_BYTETYPE local_cb = one_ycbcr.at(1);
    YCRCB_BYTETYPE local_cr = one_ycbcr.at(2);
    // LOG(INFO) << "pass one_ycbcr";

    // 111 110 101 100  011 010 001 000
    //  7   6   5    4    3   2  1  0 
    //  -4  -3  -2   -1   0   1  2  3
    uint8_t v = cluster_center_index;
    v = v << 6;
    int cb_delta = (int) local_cb - (int) base_cb;
    if (cb_delta > 3) {
        cb_delta = 3;
    } else if (cb_delta < -4) {
        cb_delta = -4;
    }
    uint8_t unsign_cb_delta = 3 - cb_delta;
    unsign_cb_delta = unsign_cb_delta << 3;
    v = v | (unsign_cb_delta & 0xFF);

    int cr_delta = (int) local_cr - (int) base_cr;
    if (cr_delta > 3) {
        cr_delta = 3;
    } else if (cr_delta < -4) {
        cr_delta = -4;
    }
    uint8_t unsign_cr_delta = 3 - cr_delta;
    v = v | (unsign_cr_delta & 0xFF);
    ostream.write(reinterpret_cast<const char*>(&v), 1);
    debug_output_file << 1 << " ";

    v = local_y;
    ostream.write(reinterpret_cast<const char*>(&v), 1);
    debug_output_file << 1 << " ";
}

void ImageCompressor::compress() {
    LOG(INFO) << "begin compress " << input_file_path << " with tif version=" << TIFFGetVersion();
    int64_t start_us = get_time_since_epoch_us();

    TIFF* tiff = TIFFOpen(input_file_path.c_str(), "r");
    if (!tiff) {
        LOG(ERROR) << "open input file for read fail. file=" << input_file_path;
        return;
    }

    int dir_cnt = 0;
    do {
        dir_cnt++;
    } while (TIFFReadDirectory(tiff));
    LOG(INFO) << "input file tiff dir_cnt=" << dir_cnt;

    uint32_t width, height;
    uint16_t samples_per_pixel, bit_per_sample;
    size_t n_pixels;
    uint32_t* raster;
    uint8_t orientation, photo_metric;
    int tiled;

    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff, TIFFTAG_PHOTOMETRIC, &photo_metric);
    TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bit_per_sample);
    TIFFGetField(tiff, TIFFTAG_ORIENTATION, &orientation);
    tiled = TIFFIsTiled(tiff);
    n_pixels = width * height;

    // the following piece of code read the whole image by pre-allocating all needed memory
    // raster = (uint32_t*) _TIFFmalloc(n_pixels * sizeof(uint32_t));
    // if (!raster) {
    //     LOG(ERROR) << "_TIFFmalloc fail to malloc size" << n_pixels * sizeof(uint32_t);
    //     return;
    // }
    // LOG(INFO) << "finish _TIFFmalloc";
    // int ret = TIFFReadRGBAImage(tiff, width, height, raster, 0);
    // if (!ret) {
    //     LOG(ERROR) << "TIFFReadRGBAImage fail";
    //     return;
    // }
    // LOG(INFO) << "finish TIFFReadRGBAImage";
    
    LOG(INFO) << "Input file: \n width=" << width << ", height=" << height
              << ", \n PHOTOMETRIC=" << (int) photo_metric << ", tiled=" << tiled
              << ", \n n_pixels=" << n_pixels << ", sizeof(raster)=" << sizeof(raster)
              << ", \n samples_per_pixel=" << samples_per_pixel << ", bit_per_sample=" << bit_per_sample
              << ", \n orientation=" << (int)orientation;

    // test whether the convert function work correctly by converting white point
    uint8_t r = 245;
    uint8_t g = 245;
    uint8_t b = 245;
    uint8_t y = get_y_from_RGB(r, g, b);
    uint8_t cb = get_cb_from_RGB(r, g, b);
    uint8_t cr = get_cr_from_RGB(r, g, b);
    LOG(INFO) << "provided: r=" << (int) r << ", g=" << (int) g << ", b=" << (int) b
              << ", calculated: y=" << (int) y << ", cb=" << (int) cb << ", cr=" << (int) cr
              << ", calculated: r=" << (int) get_r_from_YCBCR(y, cb, cr)
              << ", g=" << (int) get_g_from_YCBCR(y, cb, cr)
              << ", b=" << (int) get_b_from_YCBCR(y, cb, cr);

    // convert RGB to YCbCr
    size_t scan_line_size = TIFFScanlineSize(tiff);
    LOG(INFO) << "TIFF scan line size=" << scan_line_size;
    void* in_buf = _TIFFmalloc(scan_line_size);

    // code needs to filter the irrelated things which may occupy most size, e.g. white point.
    // since we still need to know which pixel belongs to which cluster center, we can not
    // let `rgb_info` store the things after filter, otherwise we do not know which pixel that
    // each element of rgb_info` is.
    // Current method: still let `rgb_info` contain all pixels and let k-means method to
    //                 filter irrelated things to avoid do clustering. The filter-ed-out
    //                 pixel will belong to cluster_id=-1 which is not the normal result of k-means


    // n_pixels * 3(R G B)
    std::vector<std::vector<double>> rgb_info;
    // n_pixels * 3(Y C_b C_r)
    std::vector<std::vector<double>> ycbcr_info;
    // n_pixels * 2(C_b C_r)
    std::vector<std::vector<double>> only_cbcr_info;

    for (int row = 0; row < height; row++) {
        int ret = TIFFReadScanline(tiff, in_buf, row, 0);

        uint32_t * rgb_array = (uint32_t *) in_buf;
        int arr_num = scan_line_size / 4;

        for (int k = 0; k < arr_num; k++) {
            uint32_t one_pixel = *(rgb_array + k);

            // contruct r/g/b
            std::vector<double> one_rgb;
            RGB_BYTETYPE r = TIFFGetR(one_pixel);
            RGB_BYTETYPE g = TIFFGetG(one_pixel);
            RGB_BYTETYPE b = TIFFGetB(one_pixel);
            one_rgb.push_back(r);
            one_rgb.push_back(g);
            one_rgb.push_back(b);
            rgb_info.push_back(one_rgb);

            // construct ycbcr
            std::vector<double> one_ycbcr;
            std::vector<double> one_only_cbcr;

            YCRCB_BYTETYPE y = get_y_from_RGB(r, g, b);
            YCRCB_BYTETYPE cb = get_cb_from_RGB(r, g, b);
            YCRCB_BYTETYPE cr = get_cr_from_RGB(r, g, b);
            
            one_ycbcr.push_back(y);
            one_ycbcr.push_back(cb);
            one_ycbcr.push_back(cr);

            one_only_cbcr.push_back(cb);
            one_only_cbcr.push_back(cr);

            ycbcr_info.push_back(one_ycbcr);
            only_cbcr_info.push_back(one_only_cbcr);

            if (row == 0 && k < 10) {
                // wrong example: read value: r=254, g=254, b=254, cal value: y=249, cb=128, cr=128, cal value: r=15, g=15, b=15
                //         calculated r by y/cb/cr is definitely not correct
                //          (1.164 * (y - 16) + 1.596 * (cr - 128)) ==> 272 ==> overflow
                //         the root cause is that the calculated Y is wrong, too big. One coefficient is too big
                LOG(INFO) << "read value: r=" << (int) r << ", g=" << (int) g << ", b=" << (int) b
                          << ", cal value: y=" << (int) y << ", cb=" << (int) cb << ", cr=" << (int) cr
                          << ", cal value: r=" << (int) get_r_from_YCBCR(y, cb, cr)
                          << ", g=" << (int) get_g_from_YCBCR(y, cb, cr)
                          << ", b=" << (int) get_b_from_YCBCR(y, cb, cr);
            }
        }
    }
    LOG(INFO) << "n_pixels=" << n_pixels << ", rgb_info.size()=" << rgb_info.size()
              << ", ycbcr_info.size()=" << ycbcr_info.size()
              << ", only_cbcr_info.size()=" << only_cbcr_info.size();

    // do clusting based on Cb and Cr
    LOG(INFO) << "do clustering";
    int point_dimision = 2;
	int point_num = only_cbcr_info.size();
    // int point_num = 30; // for debug
    int cluster_num = 3;
    KMEANS<double> kms(cluster_num, point_dimision, point_num, only_cbcr_info);
	kms.randCent();
	kms.kmeans();
    std::vector<std::vector<double>> clustering_center_points = kms.get_centroids();
    std::vector<tNode> points_cluster_info = kms.get_clusterAssment();
    LOG(INFO) << "with " << cluster_num << "-means algorithm, all cluster center point are=" << (nlohmann::json) clustering_center_points
              << ", cluster center distribution(each cluster contains how many points)=" << (nlohmann::json) kms.get_centroids_num();

    // output txt result to manually check whether the output is similar to the original image
    std::ofstream text_output_file(text_output_file_path);
    if (!text_output_file) {
        std::cout << "Could not open the output file: " << text_output_file_path << std::endl;
        std::abort();
    }
    int line_count = 0;
    for (int i = 0; i < point_num; i++) {  
        tNode info = points_cluster_info.at(i);
        char c;
        if (info.minIndex == -1) {
            c = ' ';
            text_output_file.write(reinterpret_cast<const char*>(&c), 1);
        } else if (info.minIndex == 0) {
            c = '*';
            text_output_file.write(reinterpret_cast<const char*>(&c), 1);
        } else if (info.minIndex == 1) {
            c = '+';
            text_output_file.write(reinterpret_cast<const char*>(&c), 1);
        } else if (info.minIndex == 2) {
            c = '-';
            text_output_file.write(reinterpret_cast<const char*>(&c), 1);
        } else {
            LOG(INFO) << "error: cluster index is " << info.minIndex;
        }
        line_count++;

        if (line_count == width) {
            //c = '\n';
            //text_output_file.write(reinterpret_cast<const char*>(&c), 1);
            text_output_file << "\n";
            line_count = 0;
        }
    }
    text_output_file.close();
    LOG(INFO) << "finish flush " << text_output_file_path;

    // output compressed byte array to file
    std::ofstream compres_output_file(compres_output_file_path, ios::binary);
    if (!compres_output_file) {
        std::cout << "Could not open the output file: " << compres_output_file_path << std::endl;
        std::abort();
    }

    int continuous_white_cnt = 0;
    for (int i = 0; i < point_num; i++) {
        // LOG(INFO) << i << ", point_num=" << point_num << ", ycbcr_info.size()="
        //           << ycbcr_info.size() << ", points_cluster_info.size()="
        //           << points_cluster_info.size()
        //           << ",  points_cluster_info.at(i).minIndex=" << points_cluster_info.at(i).minIndex;
        std::vector<double> one_ycbcr = ycbcr_info.at(i);

        int cluster_center_index = points_cluster_info.at(i).minIndex;

        if (cluster_center_index == -1) {
            continuous_white_cnt++;
            if (continuous_white_cnt == 63) {
                encode_white_pixels(compres_output_file, continuous_white_cnt);
                continuous_white_cnt = 0;
            }
        } else {
            if (continuous_white_cnt > 0) {
                encode_white_pixels(compres_output_file, continuous_white_cnt);
                continuous_white_cnt = 0;
            }

            encode_cluster_pixels(compres_output_file, cluster_center_index, clustering_center_points, one_ycbcr, 1);
        }
    }
    // handle the tail if white pixel does not reach to 63
    if (continuous_white_cnt > 0) {
        encode_white_pixels(compres_output_file, continuous_white_cnt);
        continuous_white_cnt = 0;
    }
    compres_output_file.close();
    LOG(INFO) << "finish flush " << compres_output_file_path;

    // read from file that contains the compressed byte array and restore the orignal image
    // based on compressed info.
    // generate the restored .tiff file
    TIFF* out_tiff = TIFFOpen(restore_output_file_path.c_str(), "w");
    if (!out_tiff) {
        LOG(ERROR) << "open file for write fail. file=" << restore_output_file_path;
        return;
    }
    LOG(INFO) << "finish TIFFOpen with output file and will generate one output file";
    debug_output_file << "\n\n\n";

    TIFFSetField(out_tiff, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(out_tiff, TIFFTAG_IMAGELENGTH, height);
    //TIFFSetField(out_tiff, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(out_tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(out_tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(out_tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    // The following two lines are important, for 28M file, without the following two line, it is only 875K. It is 32x.
    TIFFSetField(out_tiff, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    TIFFSetField(out_tiff, TIFFTAG_BITSPERSAMPLE, bit_per_sample);

    // read from compressed binary file and restore the original image
    std::filesystem::path compres_input_file_path = compres_output_file_path;
    std::ifstream compres_input_file(compres_input_file_path, ios::binary);
    if (!compres_input_file) {
        std::cout << "Could not open the input file: " << compres_input_file_path << std::endl;
        std::abort();
    }
    RGB_BYTETYPE* one_line_out_buf = (RGB_BYTETYPE *)_TIFFmalloc(scan_line_size);
    //std::memset(one_line_out_buf, 0, scan_line_size);
    //void* one_line_out_buf_to_flush = nullptr;

    RGB_BYTETYPE no_matter_a = 255;
    RGB_BYTETYPE white_r = get_r_from_YCBCR(234, 128, 128);
    RGB_BYTETYPE white_g = get_g_from_YCBCR(234, 128, 128);
    RGB_BYTETYPE white_b = get_b_from_YCBCR(234, 128, 128);

    // should flush 
    int real_time_pixel_cnt = 0;
    int one_line_pixel_num = scan_line_size / 4;
    int row = 0;

    while (!compres_input_file.eof()) {
        uint8_t one_input_byte = 0U;
        compres_input_file.read(reinterpret_cast<char*>(&one_input_byte), 1);

        int cluster_center_index = one_input_byte >> 6;
        //LOG(INFO) << "cluster_center_index = " << cluster_center_index;

        if (cluster_center_index == 3) {
            int continuous_white_cnt = one_input_byte & 0x3F;
            debug_output_file << continuous_white_cnt << " ";

            int cnt = 1;
            while (cnt <= continuous_white_cnt) {
                if (real_time_pixel_cnt == one_line_pixel_num) {
                    TIFFWriteScanline(out_tiff, one_line_out_buf, row, 0);

                    row++;
                    real_time_pixel_cnt = 0;
                }

                // r/g/b/a or a/b/g/r
                *(one_line_out_buf + real_time_pixel_cnt * 4) = white_r;
                *(one_line_out_buf + real_time_pixel_cnt * 4 + 1) = white_g;
                *(one_line_out_buf + real_time_pixel_cnt * 4 + 2) = white_b;
                *(one_line_out_buf + real_time_pixel_cnt * 4 + 3) = no_matter_a;

                if (row == 0 && real_time_pixel_cnt < 10) {
                    LOG(INFO) << (int) white_r << "\t " << (char) white_g << "\t" << (int) white_b;
                }

                // *(one_line_out_buf + real_time_pixel_cnt * 4) = no_matter_a;
                // *(one_line_out_buf + real_time_pixel_cnt * 4 + 1) = white_b;
                // *(one_line_out_buf + real_time_pixel_cnt * 4 + 2) = white_g;
                // *(one_line_out_buf + real_time_pixel_cnt * 4 + 3) = white_r;

                real_time_pixel_cnt++;

                cnt++;
            }
        } else {
            uint8_t local_y = 0U;
            compres_input_file.read(reinterpret_cast<char*>(&local_y), 1);

            debug_output_file << 1 << " ";
            debug_output_file << 1 << " ";

            if (real_time_pixel_cnt == one_line_pixel_num) {
                TIFFWriteScanline(out_tiff, one_line_out_buf, row, 0);

                row++;
                real_time_pixel_cnt = 0;
            }

            YCRCB_BYTETYPE base_cb = clustering_center_points.at(cluster_center_index).at(0);
            YCRCB_BYTETYPE base_cr = clustering_center_points.at(cluster_center_index).at(1);

            // TODO: not sure
            int cb_delta = 3 - ((one_input_byte >> 3) & 0x7);
            int cr_delta = 3 - (one_input_byte & 0x7);
            
            YCRCB_BYTETYPE local_cb = (YCRCB_BYTETYPE) ((int) base_cb + cb_delta);
            YCRCB_BYTETYPE local_cr = (YCRCB_BYTETYPE) ((int) base_cr + cr_delta);

            RGB_BYTETYPE local_r = get_r_from_YCBCR(local_y, local_cb, local_cr);
            RGB_BYTETYPE local_g = get_g_from_YCBCR(local_y, local_cb, local_cr);
            RGB_BYTETYPE local_b = get_b_from_YCBCR(local_y, local_cb, local_cr);

            // r/g/b/a or a/b/g/r
            *(one_line_out_buf + real_time_pixel_cnt * 4) = local_r;
            *(one_line_out_buf + real_time_pixel_cnt * 4 + 1) = local_g;
            *(one_line_out_buf + real_time_pixel_cnt * 4 + 2) = local_b;
            *(one_line_out_buf + real_time_pixel_cnt * 4 + 3) = no_matter_a;


            if (row == 0 && real_time_pixel_cnt < 10) {
                LOG(INFO) << (int) local_r << "\t " << (int) local_g << "\t" << (int) local_b;
            }
            // *(one_line_out_buf + real_time_pixel_cnt * 4) = no_matter_a;
            // *(one_line_out_buf + real_time_pixel_cnt * 4 + 1) = local_b;
            // *(one_line_out_buf + real_time_pixel_cnt * 4 + 2) = local_g;
            // *(one_line_out_buf + real_time_pixel_cnt * 4 + 3) = local_r;

            real_time_pixel_cnt++;
        }
    }
    if (real_time_pixel_cnt != one_line_pixel_num) {
        LOG(ERROR) << "should have a complete row. row=" << row
                  << ", real_time_pixel_cnt=" << real_time_pixel_cnt
                  << ", one_line_pixel_num=" << one_line_pixel_num;
    }
    TIFFWriteScanline(out_tiff, one_line_out_buf, row, 0);

    compres_input_file.close();
    LOG(INFO) << "restore image finish. row=" << row << ", original height=" << height;
    if (row != height) {
        LOG(ERROR) << "restored image should have same lines with origin image";
    }

    LOG(INFO) << "compress used time(ms)=" << (get_time_since_epoch_us() - start_us) / 1000;

    TIFFClose(out_tiff);
    TIFFClose(tiff);

    // _TIFFfree(raster);

    // read file
    // convert to one array
    // K-means
    // output file
}

void ImageCompressor::copy_and_output_file() {
    LOG(INFO) << "being handle input file " << input_file_path << " with tif version=" << TIFFGetVersion();

    TIFF* tiff = TIFFOpen(input_file_path.c_str(), "r");
    if (!tiff) {
        LOG(ERROR) << "open input file for read fail. file=" << input_file_path;
        return;
    }

    int dir_cnt = 0;
    do {
        dir_cnt++;
    } while (TIFFReadDirectory(tiff));
    LOG(INFO) << "input file tiff dir_cnt=" << dir_cnt;

    uint32_t width, height;
    uint16_t samples_per_pixel, bit_per_sample;
    size_t n_pixels;
    uint32_t* raster;
    uint8_t orientation, photo_metric;
    int tiled;

    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff, TIFFTAG_PHOTOMETRIC, &photo_metric);
    TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bit_per_sample);
    TIFFGetField(tiff, TIFFTAG_ORIENTATION, &orientation);
    tiled = TIFFIsTiled(tiff);
    n_pixels = width * height;

    // the following piece of code read the whole image by pre-allocating all needed memory
    // raster = (uint32_t*) _TIFFmalloc(n_pixels * sizeof(uint32_t));
    // if (!raster) {
    //     LOG(ERROR) << "_TIFFmalloc fail to malloc size" << n_pixels * sizeof(uint32_t);
    //     return;
    // }
    // LOG(INFO) << "finish _TIFFmalloc";
    // int ret = TIFFReadRGBAImage(tiff, width, height, raster, 0);
    // if (!ret) {
    //     LOG(ERROR) << "TIFFReadRGBAImage fail";
    //     return;
    // }
    // LOG(INFO) << "finish TIFFReadRGBAImage";
    
    LOG(INFO) << "Input file: \n width=" << width << ", height=" << height
              << ", \n PHOTOMETRIC=" << (int) photo_metric << ", tiled=" << tiled
              << ", \n n_pixels=" << n_pixels << ", sizeof(raster)=" << sizeof(raster)
              << ", \n samples_per_pixel=" << samples_per_pixel << ", bit_per_sample=" << bit_per_sample
              << ", \n orientation=" << (int)orientation;

    std::filesystem::path output_file = base_path / "out_by_copy.tif";
    TIFF* out_tiff = TIFFOpen(output_file.c_str(), "w");
    if (!out_tiff) {
        LOG(ERROR) << "open file for write fail. file=" << output_file;
        return;
    }
    LOG(INFO) << "finish TIFFOpen with output file and will generate one output file";

    // the following two arguments can amplify the input image with (width_multiple * height_multiple)
    int width_multiple = 1;
    int height_multiple = 1;

    TIFFSetField(out_tiff, TIFFTAG_IMAGEWIDTH, width * width_multiple);
    TIFFSetField(out_tiff, TIFFTAG_IMAGELENGTH, height * height_multiple);
    //TIFFSetField(out_tiff, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(out_tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(out_tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(out_tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(out_tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    // The following two lines are important, for 28M file, without the following two line, it is only 875K. It is 32x.
    TIFFSetField(out_tiff, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    TIFFSetField(out_tiff, TIFFTAG_BITSPERSAMPLE, bit_per_sample);

    size_t scan_line_size = TIFFScanlineSize(tiff);
    LOG(INFO) << "TIFF scan line size=" << scan_line_size;
    void* in_buf = _TIFFmalloc(scan_line_size);
    void* out_buf = _TIFFmalloc(scan_line_size * width_multiple);
    for (int j = 0; j < height_multiple; j++) {

        int64_t start_us = get_time_since_epoch_us();

        for (int row = 0; row < height; row++) {
            int ret = TIFFReadScanline(tiff, in_buf, row, 0);
            // LOG(INFO) << "row=" << row << ", ret=" << ret << ", in_buf=" << (char *) in_buf;

            for (int i = 0; i < width_multiple; i++) {
                std::memcpy(((uint8_t*)out_buf) + i * scan_line_size, in_buf, scan_line_size);
            }
            
            TIFFWriteScanline(out_tiff, out_buf, row + j * height, 0);
            //TIFFFlush(out_tiff);
        }

        LOG(INFO) << "finish current height_multiple=" << j << ", used_time(ms)="
                  << (get_time_since_epoch_us() - start_us) / 1000;
    }
   
    TIFFClose(out_tiff);
    TIFFClose(tiff);
}

